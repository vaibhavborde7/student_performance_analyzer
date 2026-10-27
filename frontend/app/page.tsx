 "use client";

import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";
import { Activity, BrainCircuit, Database, FileUp, GraduationCap, RefreshCw, Sparkles } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type Analysis = {
  dataset: {
    rows: number; columns_count: number; predictor_count: number; target: string;
    target_mean: number; target_median: number; target_std: number; missing_cells: number;
    train_rows: number; test_rows: number; excluded_columns: string[]; predictors: string[];
    columns: { name: string; dtype: string; missing: number; unique: number; sample: unknown[] }[];
  };
  models: { name: string; r2: number; mae: number; rmse: number; cv_r2_mean: number | null; cv_r2_std: number | null }[];
  best_model: string;
  feature_importance: { feature: string; importance: number }[];
  correlations: { feature: string; correlation: number }[];
  subject_summary: { subject: string; average: number; min: number; max: number }[];
  distribution: { range: string; count: number }[];
  predictions: { row: number; actual: number; predicted: number; residual: number }[];
};

function metric(value: number | null | undefined, digits = 3) {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}
function pct(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(1)}%`;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("Average_Score");
  const [data, setData] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [filename, setFilename] = useState("");

  async function runAnalysis(chosen?: File) {
    const upload = chosen || file;
    if (!upload) { setError("Choose a CSV file first."); return; }
    setBusy(true); setError("");
    const form = new FormData();
    form.append("file", upload);
    form.append("target", target);
    try {
      const response = await fetch(`${API}/api/analyze?target=${encodeURIComponent(target)}`, {
        method: "POST", body: form
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Analysis failed.");
      setData(result);
      setFilename(upload.name);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not connect to the API. Is FastAPI running?");
    } finally { setBusy(false); }
  }

  const best = useMemo(() => data?.models.find(m => m.name === data.best_model), [data]);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-icon"><GraduationCap size={22}/></div><div><b>StudentLens</b><small>ML ANALYTICS</small></div></div>
        <div className="nav-label">WORKSPACE</div>
        <a className="nav active"><Activity size={17}/> Overview</a>
        <a className="nav"><BrainCircuit size={17}/> Model laboratory</a>
        <a className="nav"><Database size={17}/> Dataset explorer</a>
        <div className="sidebar-bottom"><div className="status-dot"/> Local ML engine <span>●</span><small>FastAPI · scikit-learn</small></div>
      </aside>

      <section className="content">
        <header className="topbar"><div><span className="eyebrow">STUDENT ANALYTICS / OVERVIEW</span><h1>Performance intelligence</h1><p>Explore student outcomes, compare models, and inspect the factors associated with scores.</p></div><div className="live-pill"><span/> MODEL WORKSPACE</div></header>

        <section className="upload-panel">
          <div className="upload-copy"><div className="icon-tile"><FileUp size={21}/></div><div><h2>Dataset workspace</h2><p>Upload a CSV to validate data, train three models, and refresh all charts.</p></div></div>
          <div className="upload-controls">
            <label className="file-picker"><input type="file" accept=".csv,text/csv" onChange={e => setFile(e.target.files?.[0] || null)}/><FileUp size={17}/><span>{file ? file.name : "Choose CSV dataset"}</span></label>
            <label className="target-field"><span>Target</span><input value={target} onChange={e => setTarget(e.target.value)} aria-label="Target column"/></label>
            <button className="primary-btn" onClick={() => runAnalysis()} disabled={busy || !file}>{busy ? <><RefreshCw className="spin" size={16}/> Training…</> : <><Sparkles size={16}/> Analyze dataset</>}</button>
          </div>
          {error && <div className="error">{error}</div>}
          {filename && !error && <div className="file-note">Last analyzed: <b>{filename}</b> · Model results are recalculated for every upload.</div>}
        </section>

        {data ? <>
          <div className="section-heading"><div><h2>Dataset snapshot</h2><p>Quick quality checks and target distribution overview.</p></div><span className="tag">TARGET · {data.dataset.target}</span></div>
          <div className="stats-grid">
            <Stat label="STUDENT RECORDS" value={data.dataset.rows.toLocaleString()} detail={`${data.dataset.columns_count} columns detected`} icon={<Database size={18}/>}/>
            <Stat label="PREDICTOR FEATURES" value={String(data.dataset.predictor_count)} detail="After leakage / ID exclusions" icon={<BrainCircuit size={18}/>}/>
            <Stat label="MEAN TARGET SCORE" value={metric(data.dataset.target_mean, 2)} detail={`Median ${metric(data.dataset.target_median, 2)}`} icon={<Activity size={18}/>}/>
            <Stat label="MISSING CELLS" value={data.dataset.missing_cells.toLocaleString()} detail="Imputed inside ML pipelines" icon={<RefreshCw size={18}/>}/>
          </div>

          <div className="section-heading"><div><h2>Model laboratory</h2><p>Held-out test metrics plus cross-validation on the training partition.</p></div><span className="tag blue">80 / 20 SPLIT</span></div>
          <div className="model-grid">
            {data.models.map((m, i) => <div className={`model-card ${m.name === data.best_model ? "winner" : ""}`} key={m.name}>
              <div className="model-top"><div className="model-symbol">{i === 0 ? "LR" : i === 1 ? "RF" : "GB"}</div>{m.name === data.best_model && <span className="best-tag">TOP TEST R²</span>}</div>
              <h3>{m.name}</h3><div className="model-r2">{pct(m.r2)} <small>R²</small></div>
              <div className="metric-row"><span>MAE</span><b>{metric(m.mae)}</b></div>
              <div className="metric-row"><span>RMSE</span><b>{metric(m.rmse)}</b></div>
              <div className="metric-row"><span>CV R² mean ± SD</span><b>{metric(m.cv_r2_mean)} ± {metric(m.cv_r2_std)}</b></div>
            </div>)}
          </div>
          <div className="chart-grid">
            <Panel title="Feature importance" subtitle="Random Forest · relative impurity-based importance">
              {data.feature_importance.length ? <ResponsiveContainer width="100%" height={280}><BarChart data={data.feature_importance.slice(0, 8)} layout="vertical" margin={{left: 18, right: 18}}><CartesianGrid strokeDasharray="3 3" stroke="#243143" horizontal={false}/><XAxis type="number" tick={{fill:"#91a0b5",fontSize:11}} axisLine={false} tickLine={false}/><YAxis type="category" dataKey="feature" width={125} tick={{fill:"#b9c5d6",fontSize:11}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#111b2a",border:"1px solid #2a3a4e",borderRadius:10}}/><Bar dataKey="importance" name="Importance" radius={[0,5,5,0]} fill="#63a8ff"/></BarChart></ResponsiveContainer> : <Empty/>}
            </Panel>
            <Panel title="Actual vs predicted" subtitle={`Best held-out model: ${data.best_model}`}>
              <ResponsiveContainer width="100%" height={280}><LineChart data={data.predictions.slice(0, 70)} margin={{top:10,right:12,bottom:5,left:0}}><CartesianGrid strokeDasharray="3 3" stroke="#243143"/><XAxis dataKey="row" tick={{fill:"#91a0b5",fontSize:10}} tickLine={false} axisLine={false} label={{value:"Test sample",position:"insideBottom",offset:-2,fill:"#91a0b5",fontSize:11}}/><YAxis tick={{fill:"#91a0b5",fontSize:10}} tickLine={false} axisLine={false}/><Tooltip contentStyle={{background:"#111b2a",border:"1px solid #2a3a4e",borderRadius:10}}/><Legend/><Line type="monotone" dataKey="actual" name="Actual" stroke="#62d6b0" dot={false} strokeWidth={2}/><Line type="monotone" dataKey="predicted" name="Predicted" stroke="#6ba9ff" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer>
            </Panel>
          </div>
          <div className="chart-grid">
            <Panel title="Target score distribution" subtitle="Count of records in score intervals">
              <ResponsiveContainer width="100%" height={250}><BarChart data={data.distribution} margin={{top:10,right:10,bottom:30,left:0}}><CartesianGrid strokeDasharray="3 3" stroke="#243143"/><XAxis dataKey="range" angle={-28} textAnchor="end" interval={0} tick={{fill:"#91a0b5",fontSize:9}} tickLine={false} axisLine={false}/><YAxis tick={{fill:"#91a0b5",fontSize:10}} tickLine={false} axisLine={false}/><Tooltip contentStyle={{background:"#111b2a",border:"1px solid #2a3a4e",borderRadius:10}}/><Bar dataKey="count" name="Students" fill="#a78bfa" radius={[5,5,0,0]}/></BarChart></ResponsiveContainer>
            </Panel>
            <Panel title="Numeric correlations" subtitle={`Correlation with ${data.dataset.target} · descriptive, not causal`}>
              {data.correlations.length ? <div className="corr-list">{data.correlations.slice(0,8).map(c => <div className="corr-item" key={c.feature}><span>{c.feature}</span><div className="corr-track"><div style={{width:`${Math.min(100,Math.abs(c.correlation)*100)}%`,background:c.correlation >= 0 ? "#62d6b0" : "#fb8b8b"}}/></div><b className={c.correlation >= 0 ? "positive" : "negative"}>{metric(c.correlation, 2)}</b></div>)}</div> : <Empty/>}
            </Panel>
          </div>
          {data.subject_summary.length > 0 && <Panel title="Subject score summary" subtitle="Descriptive statistics only; subject marks are excluded from Average_Score prediction.">
            <div className="table-wrap"><table><thead><tr><th>SUBJECT</th><th>MEAN</th><th>MIN</th><th>MAX</th></tr></thead><tbody>{data.subject_summary.map(s => <tr key={s.subject}><td>{s.subject}</td><td>{metric(s.average, 1)}</td><td>{metric(s.min, 1)}</td><td>{metric(s.max, 1)}</td></tr>)}</tbody></table></div>
          </Panel>}
          <Panel title="Data quality inspector" subtitle="Column-level null counts, unique values, and sample values.">
            <div className="table-wrap"><table><thead><tr><th>COLUMN</th><th>TYPE</th><th>MISSING</th><th>UNIQUE</th><th>SAMPLE</th></tr></thead><tbody>{data.dataset.columns.map(c => <tr key={c.name}><td>{c.name}</td><td><span className="type-pill">{c.dtype}</span></td><td>{c.missing}</td><td>{c.unique}</td><td className="sample-cell">{c.sample.map(String).join(", ") || "—"}</td></tr>)}</tbody></table></div>
          </Panel>
          <div className="footnote">Interpretation note: feature importance and correlations describe this dataset; they do not establish causation. Compare test and cross-validation metrics before selecting a model.</div>
        </> : <div className="empty-state"><div className="empty-icon"><BrainCircuit size={32}/></div><h2>Ready to train your models</h2><p>Upload your CSV, set the numeric target column, and run the pipeline. Model metrics, feature analysis, and dataset charts will appear here.</p><div className="pipeline"><span>01 · VALIDATE</span><i>→</i><span>02 · PREPROCESS</span><i>→</i><span>03 · TRAIN 3 MODELS</span><i>→</i><span>04 · EVALUATE</span></div></div>}
        <footer>STUDENTLENS <span>·</span> LOCAL MACHINE LEARNING WORKSPACE</footer>
      </section>
    </main>
  );
}

function Stat({label, value, detail, icon}: {label:string; value:string; detail:string; icon:React.ReactNode}) {
  return <div className="stat-card"><div className="stat-head"><span>{label}</span><div className="stat-icon">{icon}</div></div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></div>;
}
function Panel({title, subtitle, children}: {title:string; subtitle:string; children:React.ReactNode}) {
  return <section className="panel"><div className="panel-head"><div><h3>{title}</h3><p>{subtitle}</p></div></div>{children}</section>;
}
function Empty() { return <div className="chart-empty">Not enough compatible data to display this analysis.</div>; }
