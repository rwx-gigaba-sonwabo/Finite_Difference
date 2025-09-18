# barrier_pricer.py

import pandas as pd
from barrier_engine import BarrierEngine

def run_case(title, *,
             s, b, r, t, x, sigma, h,
             optionflag, directionflag, in_out_flag,
             k=0.0,
             barrier_status=None,
             rebate_timing_in=None,
             rebate_timing_out=None):
    eng = BarrierEngine(
        s=s, b=b, r=r, t=t, x=x, sigma=sigma, h=h,
        optionflag=optionflag, directionflag=directionflag, in_out_flag=in_out_flag,
        k=k, barrier_status=barrier_status,
        rebate_timing_in=rebate_timing_in,
        rebate_timing_out=rebate_timing_out
    )

    rows = []
    rows += list(eng.get_elements().items())   # x1,x2,y1,y2,z,mu,lambda
    rows += list(eng.get_factors().items())    # A...F
    rows += [
        ("Vanilla (A)", eng.vanilla()),
        (f"Price [{optionflag.upper()}|{directionflag.upper()}|{in_out_flag.upper()}|"
         f"k={k}|status={barrier_status}|rin={rebate_timing_in}|rout={rebate_timing_out}]",
         eng.price())
    ]

    df = pd.DataFrame(rows, columns=["Component", "Value"])
    print(title)
    print(df.to_string(index=False))
    print("\n" + "-"*100 + "\n")

if __name__ == "__main__"
    base = dict(s=20.7860094, b=0.049411344, r=0.070201156, t=(174/365), x=20, sigma=0.11072180)

    # Your requested call style:
    # rebate: "Pay at Hit" or "Pay at Expiry"
    run_case("Up&Out Call (not crossed, rebate at hit)",
             **base, h=19.5, optionflag='p', directionflag='d', in_out_flag='i',
             k=(0), barrier_status='not_crossed',
             rebate_timing_out="hit")