Attribute VB_Name = "Curve_LinearRT_Pillars"
Option Explicit

' ===========================================================================
' Linear RT curve interpolation on PILLAR dates - matched to the existing
' "Curves" sheet block layout used by GetNacaRate / GetDiscountFactor:
'
'   Row 1 headers per curve block:  YieldCurve | StartDate | Date | NACA
'   Row 2 downward:                 curve name | value date| pillar | rate
'
'   col   = column where Cells(1,col)="YieldCurve" and Cells(2,col)=CurveName
'   col+1 = StartDate (curve anchor / value date)
'   col+2 = pillar Date
'   col+3 = NACA rate (decimal)
'
' Mechanics (RiskFlow 'LinearRT'):
'   1. Convert each pillar NACA to NACC:  r_nacc = Ln(1 + r_naca + bump)
'   2. Pillar tenor t_i = (PillarDate - val_date)/365      (ACT_365)
'   3. Store rt_i = r_nacc_i * t_i
'   4. Query tenor tau = (lookupDate - val_date)/365:
'        - clip tau to [t_min, t_max]
'        - linear interpolation of rt at the clipped tenor
'        - rescale by tau/tau_clipped  (flat-RATE extrapolation beyond ends)
'   5. DF = Exp(-rt(tau))
'
' Public entry points (mirroring the existing function family):
'   GetDiscountFactorRT(val_date, lookupDate, CurveName, bump)   -> DF
'   GetNaccRateRT(val_date, lookupDate, CurveName, bump)         -> NACC zero
'   GetNacaRateRT(val_date, lookupDate, CurveName, bump)         -> NACA zero
' ===========================================================================


' ---------------------------------------------------------------------------
' Same navigation as the existing workbook: scan row 1 for "YieldCurve"
' headers and match the curve name in row 2 of that column.
' ---------------------------------------------------------------------------
Function GetCurveColumnRT(CurveName As String) As Variant
    Dim lastCol As Long
    Dim col As Long
    Dim ws As Worksheet

    Set ws = ThisWorkbook.Worksheets("Curves")

    lastCol = ws.Cells(1, ws.Columns.count).End(xlToLeft).Column

    For col = 1 To lastCol
        If ws.Cells(1, col).Value = "YieldCurve" And ws.Cells(2, col).Value = CurveName Then
            GetCurveColumnRT = col
            Exit Function
        End If
    Next col
    GetCurveColumnRT = CVErr(xlErrNA)
End Function


' ---------------------------------------------------------------------------
' Core Linear RT evaluator: returns r_nacc(tau)*tau for the query tenor,
' reading the pillar block directly off the Curves sheet.
' Returns True on success, False on lookup failure.
' ---------------------------------------------------------------------------
Private Function EvalRT_FromSheet(ByVal val_date As Date, _
                                   ByVal lookupDate As Date, _
                                   ByVal CurveName As String, _
                                   ByVal bump As Double, _
                                   ByRef rtOut As Double) As Boolean
    Dim ws As Worksheet
    Dim lastrow As Long
    Dim col As Variant
    Dim i As Long, n As Long

    EvalRT_FromSheet = False

    Set ws = ThisWorkbook.Worksheets("Curves")

    col = GetCurveColumnRT(CurveName)
    If IsError(col) Then Exit Function

    lastrow = ws.Cells(ws.Rows.count, col + 2).End(xlUp).Row
    If lastrow < 2 Then Exit Function

    ' --- load pillar tenors and NACC r*t values ---
    Dim tenors() As Double, rts() As Double
    ReDim tenors(1 To lastrow - 1)
    ReDim rts(1 To lastrow - 1)

    Dim t As Double, rNaca As Double, rNacc As Double
    n = 0
    For i = 2 To lastrow
        t = (CDate(ws.Cells(i, col + 2).Value) - val_date) / 365#
        If t > 0 Then
            rNaca = CDbl(ws.Cells(i, col + 3).Value) + bump
            rNacc = Log(1# + rNaca)
            n = n + 1
            tenors(n) = t
            rts(n) = rNacc * t
        End If
    Next i

    If n = 0 Then Exit Function

    ' --- defensive insertion sort by tenor (sheet usually already sorted) ---
    Dim j As Long, kT As Double, kRT As Double
    For i = 2 To n
        kT = tenors(i): kRT = rts(i)
        j = i - 1
        Do While j >= 1
            If tenors(j) > kT Then
                tenors(j + 1) = tenors(j): rts(j + 1) = rts(j)
                j = j - 1
            Else
                Exit Do
            End If
        Loop
        tenors(j + 1) = kT: rts(j + 1) = kRT
    Next i

    ' --- query tenor ---
    Dim tau As Double, tauC As Double
    tau = (lookupDate - val_date) / 365#
    If tau <= 0 Then
        rtOut = 0#
        EvalRT_FromSheet = True
        Exit Function
    End If

    ' RiskFlow clip to pillar range
    tauC = tau
    If tauC < tenors(1) Then tauC = tenors(1)
    If tauC > tenors(n) Then tauC = tenors(n)

    ' linear interpolation in r*t space at the clipped tenor
    Dim rtC As Double
    If n = 1 Then
        rtC = rts(1)
    Else
        Dim lo As Long, hi As Long, mid As Long
        lo = 1: hi = n
        Do While hi - lo > 1
            mid = (lo + hi) \ 2
            If tenors(mid) <= tauC Then lo = mid Else hi = mid
        Loop
        Dim alpha As Double
        alpha = (tauC - tenors(lo)) / (tenors(hi) - tenors(lo))
        rtC = rts(lo) * (1# - alpha) + rts(hi) * alpha
    End If

    ' RiskFlow 'RT' rescale: tau / clipped(tau)  => flat-rate extrapolation
    rtOut = rtC * (tau / tauC)
    EvalRT_FromSheet = True
End Function


' ===========================================================================
' PUBLIC FUNCTIONS
' ===========================================================================

' Discount factor from val_date to lookupDate via Linear RT interpolation.
' Direct pillar-curve replacement for the existing GetDiscountFactor.
Public Function GetDiscountFactorRT(val_date As Date, lookupDate As Date, _
                                     CurveName As String, bump As Double) As Variant
    Dim rt As Double

    If Not EvalRT_FromSheet(val_date, lookupDate, CurveName, bump, rt) Then
        GetDiscountFactorRT = CVErr(xlErrNA)
        Exit Function
    End If

    GetDiscountFactorRT = Exp(-rt)
End Function


' Interpolated NACC zero rate at lookupDate:  r_nacc = rt(tau)/tau
Public Function GetNaccRateRT(val_date As Date, lookupDate As Date, _
                               CurveName As String, bump As Double) As Variant
    Dim rt As Double
    Dim tau As Double

    tau = (lookupDate - val_date) / 365#
    If tau <= 0 Then
        GetNaccRateRT = CVErr(xlErrNum)
        Exit Function
    End If

    If Not EvalRT_FromSheet(val_date, lookupDate, CurveName, bump, rt) Then
        GetNaccRateRT = CVErr(xlErrNA)
        Exit Function
    End If

    GetNaccRateRT = rt / tau
End Function


' Interpolated NACA zero rate at lookupDate:  r_naca = Exp(r_nacc) - 1
' Keeps compatibility with the existing NACA-based downstream functions
' (e.g. GetForwardRateNACA / GetForwardRateNACC use (1+naca)^t compounding).
Public Function GetNacaRateRT(val_date As Date, lookupDate As Date, _
                               CurveName As String, bump As Double) As Variant
    Dim nacc As Variant

    nacc = GetNaccRateRT(val_date, lookupDate, CurveName, bump)
    If IsError(nacc) Then
        GetNacaRateRT = nacc
        Exit Function
    End If

    GetNacaRateRT = Exp(CDbl(nacc)) - 1#
End Function


' Tenor-based (re-anchored) DF for the RiskFlow CPI forecast mechanic:
' evaluates the val_date curve at raw tenor (EndDate - StartDate), i.e.
'   DF_fwd = Exp( -r(tau)*tau ),  tau = (EndDate - StartDate)/365
' Use with StartDate = last CPI publication date, EndDate = sample month:
'   CPI(sample) = CPI(lastpub) / GetDiscountFactorRT_Tenor(...)
Public Function GetDiscountFactorRT_Tenor(val_date As Date, StartDate As Date, _
                                           EndDate As Date, CurveName As String, _
                                           bump As Double) As Variant
    Dim tauDays As Long
    Dim rt As Double

    tauDays = EndDate - StartDate
    If tauDays <= 0 Then
        GetDiscountFactorRT_Tenor = 1#
        Exit Function
    End If

    ' re-anchored query: same as looking up val_date + tauDays on the
    ' val_date-anchored curve
    If Not EvalRT_FromSheet(val_date, val_date + tauDays, CurveName, bump, rt) Then
        GetDiscountFactorRT_Tenor = CVErr(xlErrNA)
        Exit Function
    End If

    GetDiscountFactorRT_Tenor = Exp(-rt)
End Function
