"""Hours 1-3: evaluation of walk-forward forecasts (MAE, RMSE, R2, actual-vs-predicted)."""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
P, OUT = sys.argv[1], sys.argv[2]
p = pd.read_csv(P, parse_dates=["StartDate"]); y = p.actual
models = [
    "Historical mean (baseline)",
    "Linear Regression (6 feats)",
    "Ridge (all feats, extra)",
    "Random Forest"
]

# Automatically identify the boosting model
boost_models = [
    col for col in p.columns
    if "boost" in col.lower() or "xgboost" in col.lower()
]

models.extend(boost_models)

print("Models being evaluated:", models)
rows=[]
for m in models:
    e=p[m]-y
    rows.append({"model":m,"MAE":mean_absolute_error(y,p[m]),"RMSE":mean_squared_error(y,p[m])**.5,
      "R2":r2_score(y,p[m]),"bias(mean err)":e.mean(),"corr(actual,pred)":np.corrcoef(y,p[m])[0,1],
      "pred_std":p[m].std(),"within_15_runs":(e.abs()<=15).mean()})
res=pd.DataFrame(rows).round(3); res.to_csv(f"{OUT}/evaluation_metrics.csv",index=False)
print(f"n={len(p)} forecasts | actual mean {y.mean():.1f} std {y.std():.1f}"); print(res.to_string(index=False))
# error by outcome band
band=pd.cut(y,[-1,9,29,59,200],labels=["0-9","10-29","30-59","60+"])
print("\nMAE by actual-runs band (n):"); print(pd.DataFrame({m.split(' (')[0]:(p[m]-y).abs().groupby(band,observed=True).mean() for m in models}).assign(n=band.value_counts().sort_index()).round(1).to_string())
# --- plots ---
B,G,INK,MUT,O="#2F6DB5","#B8BEC6","#1F2933","#6B7280","#C0562F"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":G,"axes.labelcolor":MUT,
 "xtick.color":MUT,"ytick.color":MUT,"axes.titleweight":"bold","axes.titlecolor":INK,"axes.titlelocation":"left"})
fig,ax=plt.subplots(2,3,figsize=(15,9.5)); ax=ax.ravel()
for a,m,(_,r) in zip(ax,models,res.iterrows()):
    a.plot([0,135],[0,135],color=G,ls="--",lw=1)
    a.scatter(y,p[m],color=O if "baseline" in m else B,s=26,alpha=.8)
    a.set(xlim=(-3,135),ylim=(-3,135),title=m,xlabel="actual runs",ylabel="predicted runs")
    a.text(.97,.04,f"MAE {r.MAE:.1f}  RMSE {r.RMSE:.1f}\nR² {r.R2:.2f}",transform=a.transAxes,ha="right",fontsize=9.5,color=INK)
ax[5].axis("off"); ax[5].text(0,.9,"Dashed line = perfect forecast.\nPoints hugging a flat band mean the\nmodel predicts ~the average whatever\nhappens; a good model tracks the line.\n\nR² is vs the mean of the 46 test\nactuals (negative = worse than\nknowing the test-set average).",va="top",color=MUT,fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/actual_vs_predicted_scatter.png",dpi=130)
fig,ax=plt.subplots(len(models),1,figsize=(14,12),sharex=True)
t=np.arange(1,len(p)+1)
for a,m,(_,r) in zip(ax,models,res.iterrows()):
    a.plot(t,y,color=G,marker="o",ms=3.5,lw=1,label="actual"); a.plot(t,p[m],color=O if "baseline" in m else B,lw=1.8,label="predicted")
    a.set_title(f"{m}   (MAE {r.MAE:.1f}, R² {r.R2:.2f})",fontsize=10); a.set_ylabel("runs")
ax[0].legend(frameon=False,ncol=2,loc="upper left"); ax[-1].set_xlabel("forecast number (chronological, 2022-05 → 2025-05)")
plt.tight_layout(); plt.savefig(f"{OUT}/actual_vs_predicted_timeseries.png",dpi=120)
