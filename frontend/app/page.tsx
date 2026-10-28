 "use client";

import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  Pie, PieChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis
} from "recharts";
import { Activity, BrainCircuit, Database, FileUp, GraduationCap, Lightbulb, RefreshCw, Sparkles } from "lucide-react";

const API = (
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

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
  relationships: {
    feature: string; kind: "numeric" | "category"; correlation: number | null;
    points: { x: number; y: number }[]; groups: { label: string; average: number; count: number }[];
  }[];
  subject_summary: { subject: string; average: number; min: number; max: number }[];
  distribution: { range: string; count: number }[];
  predictions: { row: number; actual: number; predicted: number; residual: number }[];
  student_insights: {
    id: string; name: string; score: number; segment: string; focus_area: string;
    focus_score: number | null; attendance: number | null; study_hours: number | null; insight: string;
  }[];
};

const SEGMENT_COLORS: Record<string, string> = {
  "Needs support": "#fb8b8b", Developing: "#f6c76b", "On track": "#6ba9ff", "High achieving": "#62d6b0"
};
const MODEL_PURPOSE: Record<string, string> = {
  "Linear Regression": "A simple, interpretable baseline that estimates a straight-line relationship between the inputs and score.",
  "Random Forest": "A collection of decision trees that can capture non-linear patterns and interactions; also supplies the feature-importance chart.",
  "Gradient Boosting": "Builds trees sequentially, with each new tree focusing on errors left by earlier trees; useful for complex patterns."
};

function correlationLabel(value: number | null) {
  if (value == null || !Number.isFinite(value)) return "Correlation unavailable";
  const strength = Math.abs(value) < 0.2 ? "very weak" : Math.abs(value) < 0.4 ? "weak" : Math.abs(value) < 0.7 ? "moderate" : "strong";
  return `${strength} ${value >= 0 ? "positive" : "negative"} association (r = ${value.toFixed(2)})`;
}

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
  const [relationshipFeature, setRelationshipFeature] = useState("");

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
      const normalizedResult: Analysis = {
        ...result,
        relationships: Array.isArray(result.relationships) ? result.relationships : [],
        student_insights: Array.isArray(result.student_insights) ? result.student_insights : []
      };
      setData(normalizedResult);
      setRelationshipFeature(normalizedResult.relationships[0]?.feature || "");
      setFilename(upload.name);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not connect to the API. Is FastAPI running?");
    } finally { setBusy(false); }
  }

  const best = useMemo(() => data?.models.find(m => m.name === data.best_model), [data]);
  const studentSegments = useMemo(() => {
    if (!data) return [];
    const counts = data.student_insights.reduce<Record<string, number>>((acc, student) => {
      acc[student.segment] = (acc[student.segment] || 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [data]);
  const focusAreas = useMemo(() => {
    if (!data) return [];
    const counts = data.student_insights.reduce<Record<string, number>>((acc, student) => {
      if (student.segment === "Needs support" || student.segment === "Developing") {
        acc[student.focus_area] = (acc[student.focus_area] || 0) + 1;
      }
      return acc;
    }, {});
    return Object.entries(counts).map(([area, students]) => ({ area, students })).sort((a, b) => b.students - a.students).slice(0, 6);
  }, [data]);
  const selectedRelationship = useMemo(
    () => data?.relationships.find(item => item.feature === relationshipFeature),
    [data, relationshipFeature]
  );

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

          {data.student_insights?.length > 0 && <>
            <div className="section-heading"><div><h2>Student insights</h2><p>Plain-language signals to help prioritize a supportive check-in.</p></div><span className="tag">{data.student_insights.length} STUDENTS GROUPED</span></div>
            <div className="chart-grid insights-charts">
              <Panel title="Class learning profile" subtitle="Students grouped by current target-score band; based on scores, not a diagnosis.">
                <div className="segment-chart"><ResponsiveContainer width="100%" height={230}><PieChart><Pie data={studentSegments} dataKey="value" nameKey="name" innerRadius={58} outerRadius={86} paddingAngle={3}>
                  {studentSegments.map(item => <Cell key={item.name} fill={SEGMENT_COLORS[item.name] || "#8e9db1"}/>)}
                </Pie><Tooltip contentStyle={{background:"#111b2a",border:"1px solid #2a3a4e",borderRadius:10}}/><Legend verticalAlign="middle" align="right" layout="vertical" wrapperStyle={{fontSize:11,color:"#b9c5d6"}}/></PieChart></ResponsiveContainer>
                <div className="segment-key">{studentSegments.map(item => <div key={item.name}><i style={{background:SEGMENT_COLORS[item.name]}}/><span>{item.name}</span><b>{item.value}</b></div>)}</div></div>
              </Panel>
              <Panel title="Common focus areas" subtitle="Most frequent lowest-scoring subject among students in the first two bands.">
                {focusAreas.length ? <ResponsiveContainer width="100%" height={270}><BarChart data={focusAreas} layout="vertical" margin={{left:10,right:18}}><CartesianGrid strokeDasharray="3 3" stroke="#243143" horizontal={false}/><XAxis type="number" allowDecimals={false} tick={{fill:"#91a0b5",fontSize:10}} axisLine={false} tickLine={false}/><YAxis type="category" dataKey="area" width={125} tick={{fill:"#b9c5d6",fontSize:11}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#111b2a",border:"1px solid #2a3a4e",borderRadius:10}}/><Bar dataKey="students" name="Students" fill="#a78bfa" radius={[0,5,5,0]}/></BarChart></ResponsiveContainer> : <Empty/>}
                <div className="insight-hint"><Lightbulb size={15}/> Use this as a conversation starter; confirm each student’s context before acting.</div>
              </Panel>
            </div>
            <Panel title="Suggested student check-ins" subtitle="Students with the lowest current scores are shown first. Suggestions are descriptive prompts for teacher review.">
              <div className="student-insight-grid">{data.student_insights.slice(0, 8).map(student => <article className="student-insight-card" key={student.id}>
                <div className="student-card-top"><div><span className="student-id">{student.id}</span><h4>{student.name}</h4></div><strong>{metric(student.score, 1)}</strong></div>
                <div className="student-card-meta"><span className={`segment-badge segment-${student.segment.toLowerCase().replaceAll(" ", "-")}`}>{student.segment}</span><span>Focus · {student.focus_area}{student.focus_score != null ? ` (${metric(student.focus_score, 0)})` : ""}</span></div>
                <p>{student.insight}</p>
                {(student.attendance != null || student.study_hours != null) && <small>{student.attendance != null ? `Attendance ${metric(student.attendance, 0)}%` : ""}{student.attendance != null && student.study_hours != null ? " · " : ""}{student.study_hours != null ? `${metric(student.study_hours, 1)} study hrs/week` : ""}</small>}
              </article>)}</div>
              <div className="insight-disclaimer">Classifications are simple score bands (60 / 75 / 85 for Average_Score; otherwise dataset quartiles). They are not labels of ability, and should not be used alone for high-stakes decisions.</div>
            </Panel>
          </>}

          <div className="section-heading"><div><h2>Model laboratory</h2><p>Held-out test metrics plus cross-validation on the training partition.</p></div><span className="tag blue">80 / 20 SPLIT</span></div>
          <div className="model-grid">
            {data.models.map((m, i) => <div className={`model-card ${m.name === data.best_model ? "winner" : ""}`} key={m.name}>
              <div className="model-top"><div className="model-symbol">{i === 0 ? "LR" : i === 1 ? "RF" : "GB"}</div>{m.name === data.best_model && <span className="best-tag">TOP TEST R²</span>}</div>
              <h3>{m.name}</h3><p className="model-purpose">{MODEL_PURPOSE[m.name]}</p><div className="model-r2" title="R-squared on the held-out test data">{pct(m.r2)} <small>R²</small></div>
              <div className="metric-row" title="Average absolute prediction error in target-score units"><span>MAE · typical error</span><b>{metric(m.mae, 2)}</b></div>
              <div className="metric-row" title="Root mean squared error; larger misses count more"><span>RMSE · penalizes big misses</span><b>{metric(m.rmse, 2)}</b></div>
              <div className="metric-row" title="R² averaged across training folds; SD reflects variation"><span>CV R² · mean ± variation</span><b>{metric(m.cv_r2_mean, 2)} ± {metric(m.cv_r2_std, 2)}</b></div>
            </div>)}
          </div>
          <div className="model-guide"><div><b>How to read these results</b><p>R² compares predictions with a baseline that always guesses the average: 1 is a perfect fit, 0 is baseline-level, and below 0 is worse than that baseline. MAE and RMSE are in the target’s units, so lower is better. RMSE penalizes occasional large errors more strongly.</p></div><div><b>Why use test and cross-validation?</b><p>The 20% held-out test set gives an independent final check. Cross-validation repeats training/validation across folds in the remaining data; a high mean with a smaller ± SD is generally more consistent. Results are dataset-dependent, not a guarantee of future performance.</p></div></div>
          <div className="section-heading"><div><h2>Compare two variables</h2><p>See how a numeric habit or category varies alongside the selected score target.</p></div><span className="tag">DESCRIPTIVE · NOT CAUSAL</span></div>
          <Panel title={selectedRelationship ? `${selectedRelationship.feature} vs ${data.dataset.target}` : "Variable relationship"} subtitle="Numeric fields are plotted as individual records; categories show average target scores by group.">
            {data.relationships.length ? <>
              <label className="relationship-select"><span>Choose a variable</span><select value={relationshipFeature} onChange={event => setRelationshipFeature(event.target.value)}>{data.relationships.map(item => <option key={item.feature} value={item.feature}>{item.feature}</option>)}</select></label>
              {selectedRelationship?.kind === "numeric" ? <>
                <div className="relationship-summary">{correlationLabel(selectedRelationship.correlation)} <span>· Pearson r measures linear association only.</span></div>
                <ResponsiveContainer width="100%" height={300}><ScatterChart margin={{top:12,right:18,bottom:22,left:4}}><CartesianGrid strokeDasharray="3 3" stroke="#243143"/><XAxis type="number" dataKey="x" name={selectedRelationship.feature} tick={{fill:"#91a0b5",fontSize:10}} axisLine={false} tickLine={false} label={{value:selectedRelationship.feature,position:"insideBottom",offset:-12,fill:"#91a0b5",fontSize:11}}/><YAxis type="number" dataKey="y" name={data.dataset.target} tick={{fill:"#91a0b5",fontSize:10}} axisLine={false} tickLine={false} label={{value:data.dataset.target,angle:-90,position:"insideLeft",fill:"#91a0b5",fontSize:11}}/><Tooltip cursor={{strokeDasharray:"3 3"}} contentStyle={{background:"#111b2a",border:"1px solid #2a3a4e",borderRadius:10}} formatter={(value: number, name: string) => [metric(value, 2), name === "x" ? selectedRelationship.feature : data.dataset.target]}/><Scatter name="Student records" data={selectedRelationship.points} fill="#6ba9ff" fillOpacity={0.65}/></ScatterChart></ResponsiveContainer>
              </> : selectedRelationship?.kind === "category" ? <>
                <div className="relationship-summary">Group averages compare the target across category values. Counts show how many records contribute to each average.</div>
                <ResponsiveContainer width="100%" height={280}><BarChart data={selectedRelationship.groups} margin={{top:14,right:18,bottom:8,left:0}}><CartesianGrid strokeDasharray="3 3" stroke="#243143" vertical={false}/><XAxis dataKey="label" tick={{fill:"#b9c5d6",fontSize:11}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#91a0b5",fontSize:10}} axisLine={false} tickLine={false} label={{value:`Mean ${data.dataset.target}`,angle:-90,position:"insideLeft",fill:"#91a0b5",fontSize:10}}/><Tooltip contentStyle={{background:"#111b2a",border:"1px solid #2a3a4e",borderRadius:10}} formatter={(value: number) => [metric(value, 2), "Mean target"]}/><Bar dataKey="average" name="Mean target" fill="#62d6b0" radius={[5,5,0,0]}/></BarChart></ResponsiveContainer>
                <div className="table-wrap"><table><thead><tr><th>GROUP</th><th>AVERAGE {data.dataset.target.toUpperCase()}</th><th>STUDENTS</th></tr></thead><tbody>{selectedRelationship.groups.map(group => <tr key={group.label}><td>{group.label}</td><td>{metric(group.average, 2)}</td><td>{group.count}</td></tr>)}</tbody></table></div>
              </> : <Empty/>}
            </> : <Empty/>}
          </Panel>
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
