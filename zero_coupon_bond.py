from datetime import date


class ZeroCouponBond:
    """
    A zero-coupon (discount) bond with a single payment at maturity.
    B = F * exp(-r * T)
    """
    def __init__(self, face_value: float, maturity_date: date):
        if face_value <= 0:
            raise ValueError("face_value must be positive.")
        if not isinstance(maturity_date, date):
            raise TypeError("maturity_date must be a datetime.date.")
        self.face_value = face_value
        self.maturity_date = maturity_date

    def __str__(self) -> str:
        return (
            f"ZeroCouponBond(face_value={self.face_value}, "
            f"maturity_date={self.maturity_date})"
        )
