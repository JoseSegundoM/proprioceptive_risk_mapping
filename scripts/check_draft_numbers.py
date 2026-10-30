"""Arithmetic checks on numbers printed in the two PDFs. Inputs are transcribed from them."""
import math
g=9.81; mL,mW=15.5,18.5; Paux=25.0; PstL,PstW=86.0,47.0
T1=[("Flat",.3,1.21,51,1.00,.29,15,1.00),("Flat",.6,.99,42,1.00,.22,11,1.00),
    ("Gravel",.3,1.38,58,1.00,.69,35,1.00),("Gravel",.6,1.12,47,1.00,.56,28,.60),
    ("Slope",.3,1.61,68,.80,.84,42,.80),("Slope",.6,1.39,59,.80,.73,37,.60),
    ("Rough",.3,1.62,69,1.00,1.41,71,.40),("Rough",.6,1.36,57,.80,1.08,54,.80)]
print("Doc B Table 1: CoT->Wh/km (printed) and implied mean power P=CoT*m*g*v [W]")
for t,v,cl,el,sl,cw,ew,sw in T1:
    wl=cl*mL*g/3.6; ww=cw*mW*g/3.6; pl=cl*mL*g*v; pw=cw*mW*g*v
    print(f"{t:7s}{v:.1f}  L {wl:5.1f}/{el:3d} Wh/km P={pl:6.1f}W (stand {PstL:.0f}) | W {ww:5.1f}/{ew:3d} Wh/km P={pw:6.1f}W (stand {PstW:.0f}, Paux {Paux:.0f}){'  <-- below standing' if pw<PstW else ''}{' and below Paux' if pw<Paux else ''}")
print("trials implied: 4 terrains*2 modes*2 speeds*5 reps =",4*2*2*5,"(paper says 92)")
print("\nDoc B Table 2: E/Ps of means vs printed")
T2=[(0,"LEG",4.3,.551,7.9),(0,"REACT",2.0,.391,5.4),(0,"E-ONLY",1.3,.539,2.6),(0,"OURS",1.6,.619,2.6),
    (.3,"LEG",4.3,.551,7.9),(.3,"REACT",3.1,.396,8.0),(.3,"E-ONLY",1.6,.477,3.5),(.3,"OURS",2.0,.572,3.6)]
for e,m,E,P,R in T2: print(f"eps={e} {m:7s} E/Ps={E/P:5.2f} printed {R}")
print("reduction E/Ps OURS vs LEG eps0: %.1f%%; E only: %.1f%%; OURS vs REACT eps.3: %.1f%%; OURS vs LEG eps.3: %.1f%%"%(100*(1-2.6/7.9),100*(1-1.6/4.3),100*(1-3.6/8.0),100*(1-3.6/7.9)))
print("LEG implied Wh/km over 94 m: %.1f"%(4.3/.094))
print("\nDoc A: binomial half-width 1.96*sqrt(p(1-p)/100) and Wilson 95% vs printed; exposure change")
def wilson(p,n,z=1.96):
    d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
for lab,p,pr in [("L geom",.81,8),("L foot",.84,7),("L prop",.92,5),("W geom",.59,10),("W foot",.70,9),("W prop",.78,8)]:
    lo,hi=wilson(p,100); print(f"{lab}: wald +-{196*math.sqrt(p*(1-p)/100):.1f} printed +-{pr}; wilson [{100*lo:.1f},{100*hi:.1f}]")
print("exposure: L prop %.1f%%, W prop %.1f%%; L foot %+.1f%%, W foot %+.1f%%"%(100*(1.46/2.86-1),100*(1.38/2.30-1),100*(2.91/2.86-1),100*(2.43/2.30-1)))
print("route overhead: L %.1f%%, W %.1f%%"%(100*(31.3/29.9-1),100*(31.1/30.0-1)))
