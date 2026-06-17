from flask import Flask, render_template_string, request, jsonify
import uuid
import time
import json
import numpy as np
import subprocess
import os
import re
from collections import defaultdict, Counter

app = Flask(__name__)
app.secret_key = 'real_cognitive_load_2024'

sessions = {}

class RealCognitiveAnalyzer:
    def __init__(self):
        self.feature_weights = {
            'keystroke_rate': 0.15,
            'error_frequency': 0.25,
            'code_complexity': 0.20,
            'run_frequency': 0.10,
            'edit_ratio': 0.15,
            'syntax_errors': 0.10,
            'runtime_errors': 0.05
        }
    
    def extract_real_features(self, session_data):
        """Extract 7 real coding behavior features"""
        keystrokes = len(session_data.get('keystrokes', []))
        total_time = time.time() - session_data['start_time']
        runs = len(session_data.get('runs', []))
        errors = len(session_data.get('errors', []))
        code_history = session_data.get('code_history', [])
        
        # 1. Keystroke rate (chars per minute)
        keystroke_rate = keystrokes / max(1, total_time / 60)
        
        # 2. Error frequency
        error_freq = errors / max(1, runs)
        
        # 3. Code complexity (lines + functions + loops)
        complexity = 0
        for code in code_history[-3:]:  # Last 3 codes
            lines = len(code.split('\n'))
            functions = len(re.findall(r'def\s+\w+', code))
            loops = len(re.findall(r'for|while', code))
            complexity += lines + functions * 5 + loops * 3
        
        # 4. Run frequency
        run_freq = runs / max(1, total_time / 60)
        
        # 5. Edit ratio (code changes)
        edit_ratio = 0
        if len(code_history) > 1:
            edit_ratio = 1 - len(set(code_history[-2:])) / max(1, len(code_history[-1]))
        
        # 6. Syntax vs runtime errors
        syntax_errors = sum(1 for e in session_data.get('errors', []) if 'SyntaxError' in str(e.get('error', '')))
        runtime_errors = errors - syntax_errors
        
        features = np.array([
            keystroke_rate, error_freq, complexity / 10, run_freq,
            edit_ratio, syntax_errors, runtime_errors
        ])
        
        return features
    
    def predict_cognitive_load(self, features):
        """Real ML-like prediction based on weighted features"""
        # Calculate cognitive load score (0-1)
        weights = np.array(list(self.feature_weights.values()))
        score = np.dot(features, weights)
        score = np.clip(score, 0, 1)
        
        if score < 0.3:
            return "low", score
        elif score < 0.7:
            return "medium", score
        else:
            return "high", score
    
    def get_session_insights(self, session_data):
        """Detailed analytics"""
        code_history = session_data.get('code_history', [])
        errors = session_data.get('errors', [])
        
        insights = {
            'total_keystrokes': len(session_data.get('keystrokes', [])),
            'unique_functions': len(set(re.findall(r'def\s+(\w+)', ' '.join(code_history)))),
            'most_common_error': Counter([e.get('error', '')[:50] for e in errors]).most_common(1)[0][0] if errors else "None",
            'avg_code_lines': np.mean([len(c.split('\n')) for c in code_history]) if code_history else 0,
            'productivity_score': len(code_history) * 10 / max(1, time.time() - session_data['start_time'])
        }
        return insights

# Global analyzer
analyzer = RealCognitiveAnalyzer()

class CodeExecutor:
    @staticmethod
    def execute_code(code):
        temp_file = f"temp_{uuid.uuid4().hex[:8]}.py"
        try:
            with open(temp_file, 'w') as f:
                f.write(code)
            
            result = subprocess.run(['python', temp_file], 
                                  capture_output=True, text=True, timeout=5)
            
            os.remove(temp_file)
            
            if result.returncode == 0:
                return {'success': True, 'output': result.stdout.strip() or '✓ Success!'}
            else:
                error = result.stderr.strip()
                return {'success': False, 'error': error or 'Runtime error'}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Timeout (5s)'}
        except Exception as e:
            try: os.remove(temp_file)
            except: pass
            return {'success': False, 'error': f'Error: {str(e)}'}

@app.route('/')
def index():
    return '''
<!DOCTYPE html><html><head><title>🚀 Real Cognitive Load IDE</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;margin:0;display:flex;align-items:center;justify-content:center;}
.container{background:#fff;max-width:500px;padding:40px;border-radius:20px;box-shadow:0 20px 40px rgba(0,0,0,0.1);text-align:center;}
h1{font-size:2.5em;margin-bottom:10px;background:linear-gradient(45deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.btn-start{background:linear-gradient(45deg,#667eea,#764ba2);color:white;border:none;padding:20px 40px;font-size:1.3em;border-radius:50px;cursor:pointer;box-shadow:0 10px 30px rgba(102,126,234,0.4);transition:all 0.3s;}
.btn-start:hover{transform:translateY(-3px);box-shadow:0 15px 40px rgba(102,126,234,0.6);}
#status{margin-top:30px;padding:20px;background:#f8f9fa;border-radius:15px;}</style></head>
<body><div class="container">
<h1>🧠 Cognitive Load IDE</h1>
<p style="color:#666;font-size:1.1em;">AI analyzes your REAL coding behavior</p>
<button class="btn-start" onclick="startSession()">🚀 Start Coding Session</button>
<div id="status"></div>
</div><script>
async function startSession(){
    const res=await fetch("/start_session",{method:"POST"});
    const data=await res.json();
    document.getElementById("status").innerHTML=
    `<div style="color:#28a745;font-size:1.2em;">
        ✅ Session Created!<br>
        <strong>${data.session_id.slice(0,8)}...</strong><br><br>
        <a href="/ide/${data.session_id}" style="background:#28a745;color:white;padding:12px 24px;border-radius:25px;text-decoration:none;font-weight:bold;">👉 Open IDE</a>
    </div>`;
}</script></body></html>'''

@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'start_time': time.time(),
        'keystrokes': [],
        'runs': [],
        'code_history': [],
        'errors': []
    }
    return jsonify({'session_id': session_id})

@app.route('/ide/<session_id>')
def ide(session_id):

    if session_id not in sessions:
        return "❌ Invalid session", 404

    css = """
:root{
--bg:#0d1117;
--editor:#0d1117;
--text:#c9d1d9;
--accent:#58a6ff;
--success:#238636;
--error:#f85149;
--border:#30363d;
}

*{margin:0;padding:0;box-sizing:border-box;}

body{
font-family:"SF Mono",Monaco,"Roboto Mono",monospace;
background:var(--bg);
color:var(--text);
overflow:hidden;
}

#app{
height:100vh;
display:grid;
grid-template-rows:auto 1fr auto;
}

#header{
background:var(--bg);
padding:16px;
border-bottom:1px solid var(--border);
display:flex;
justify-content:space-between;
align-items:center;
}

#stats{
display:flex;
gap:24px;
font-size:14px;
color:#8b949e;
}

.stat{
display:flex;
flex-direction:column;
align-items:center;
background:#161b22;
padding:8px 16px;
border-radius:8px;
}

#workspace{
display:flex;
height:100%;
}

#editor{
flex:1;
height:100%;
}

#output{
width:35%;
background:#0d1117;
padding:20px;
border-left:1px solid var(--border);
overflow-y:auto;
font-family:monospace;
white-space:pre-wrap;
}

#toolbar{
display:flex;
gap:12px;
padding:16px;
background:#161b22;
border-top:1px solid var(--border);
}

.btn{
padding:10px 20px;
border:none;
border-radius:6px;
cursor:pointer;
font-weight:600;
font-size:14px;
}

.btn-run{background:var(--success);color:white;}
.btn-analyze{background:var(--accent);color:white;}
.btn-dashboard{background:#bc8cff;color:black;}

.output-success{color:var(--success);}
.output-error{color:var(--error);}
"""

    js = f"""

const sessionId = "{session_id}";
let startTime = Date.now();
let runs = 0;
let errors = 0;

let editor;

require.config({{
paths: {{
vs: "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs"
}}
}});

require(["vs/editor/editor.main"], function () {{

editor = monaco.editor.create(
document.getElementById("editor"),
{{
value:`# Write Python code here
print("🚀 Real analytics active!")

x=[1,2,3]
print(sum(x))
`,
language:"python",
theme:"vs-dark",

automaticLayout:true,
lineNumbers:"on",
autoIndent:"full",
tabSize:4,
insertSpaces:true,

fontSize:14,
minimap:{{enabled:false}},
fontFamily:"JetBrains Mono"
}}
);

/* Behavior tracking */
editor.onDidChangeModelContent((event)=>{{
fetch("/keystroke",{{
method:"POST",
headers:{{"Content-Type":"application/json"}},
body:JSON.stringify({{
session_id:sessionId,
timestamp:Date.now(),
type:"input"
}})
}});
}});

}});


/* Timer */
setInterval(()=>{{
const elapsed=Math.floor((Date.now()-startTime)/1000);
document.getElementById("time").textContent=
String(Math.floor(elapsed/60)).padStart(2,"0")+":"+
String(elapsed%60).padStart(2,"0");
}},1000);



async function runCode(){{
const code=editor.getValue();

document.getElementById("output").textContent="⏳ Executing...";

try{{

const res=await fetch("/execute_pro",{{
method:"POST",
headers:{{"Content-Type":"application/json"}},
body:JSON.stringify({{
session_id:sessionId,
code:code
}})
}});

const data=await res.json();

runs++;
document.getElementById("runs").textContent=runs;

if(data.success){{

const out=data.output
.replace(/&/g,"&amp;")
.replace(/</g,"&lt;")
.replace(/>/g,"&gt;");

document.getElementById("output").innerHTML=
`<div class="output-success">✅ Success</div><div>${{out}}</div>`;

}}
else{{

errors++;
document.getElementById("errors").textContent=errors;

const err=data.error
.replace(/&/g,"&amp;")
.replace(/</g,"&lt;")
.replace(/>/g,"&gt;");

document.getElementById("output").innerHTML=
`<div class="output-error">❌ Error</div><div>${{err}}</div>`;

}}

}}
catch(e){{
document.getElementById("output").innerHTML=
'<div class="output-error">Network error</div>';
}}

}}



function analyze(){{
window.open("/analyze/"+sessionId,"_blank");
}}

function dashboard(){{
window.open("/dashboard/"+sessionId,"_blank");
}}
"""

    html = f"""
<!DOCTYPE html>
<html>

<head>
<title>Python IDE</title>

<meta name="viewport" content="width=device-width,initial-scale=1">

<style>{css}</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs/loader.min.js"></script>

</head>

<body>

<div id="app">

<header id="header">

<div style="font-size:18px;font-weight:600;">🐍 Python IDE</div>

<div id="stats">

<div class="stat">
<span id="time">00:00</span>
<span>Time</span>
</div>

<div class="stat">
<span id="runs">0</span>
<span>Runs</span>
</div>

<div class="stat">
<span id="errors">0</span>
<span>Errors</span>
</div>

</div>

</header>


<div id="workspace">

<div id="editor"></div>

<div id="output">
Ready to execute... 👈 Write code & click Run
</div>

</div>


<div id="toolbar">

<button class="btn btn-run" onclick="runCode()">▶ Run Code</button>

<button class="btn btn-analyze" onclick="analyze()">🧠 Analyze Load</button>

<button class="btn btn-dashboard" onclick="dashboard()">📊 Dashboard</button>

</div>

</div>


<script>
{js}
</script>

</body>

</html>
"""

    return html
    
@app.route('/execute_pro', methods=['POST'])
def execute_pro():
    data = request.json
    session_id = data.get('session_id')
    code = data.get('code', '')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    # Update session
    sessions[session_id]['code_history'].append(code)
    sessions[session_id]['runs'].append({'timestamp': time.time()})
    
    temp_file = f"temp_pro_{uuid.uuid4().hex[:8]}.py"
    start_time = time.time()
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 30s timeout for large code
        result = subprocess.run(['python', temp_file], 
                              capture_output=True, text=True, 
                              timeout=30, encoding='utf-8')
        
        execution_time = time.time() - start_time
        
        os.remove(temp_file)
        
        if result.returncode == 0:
            output = result.stdout.strip() or '✓ Code executed successfully!'
            return jsonify({
                'success': True, 
                'output': output,
                'execution_time': f"{execution_time:.2f}"
            })
        else:
            error = result.stderr.strip() or 'Runtime error'
            return jsonify({'success': False, 'error': error})
            
    except subprocess.TimeoutExpired:
        try: os.remove(temp_file)
        except: pass
        return jsonify({'success': False, 'error': '⏰ Timeout (30s) - Code too slow/large'})
    except Exception as e:
        try: os.remove(temp_file)
        except: pass
        return jsonify({'success': False, 'error': f'Executor error: {str(e)}'})
    
@app.route('/keystroke', methods=['POST'])
def log_keystroke():
    data = request.json
    session_id = data.get('session_id')
    if session_id in sessions:
        sessions[session_id]['keystrokes'].append({
            'timestamp': data.get('timestamp'),
            'key': data.get('key', ''),
            'type': data.get('type', 'input')
        })
    return jsonify({'status': 'ok'})

@app.route('/analyze/<session_id>')
def analyze(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = sessions[session_id]
    features = analyzer.extract_real_features(session_data)
    prediction, confidence = analyzer.predict_cognitive_load(features)
    insights = analyzer.get_session_insights(session_data)
    
    return jsonify({
        'prediction': prediction,
        'confidence': float(confidence),
        'load_score': float(confidence),
        'features': {k: float(v) for k, v in zip(['keystroke_rate', 'error_freq', 'complexity', 'run_freq', 'edit_ratio', 'syntax_errors', 'runtime_errors'], features)},
        'insights': insights
    })

@app.route('/dashboard/<session_id>')
def dashboard(session_id):
    if session_id not in sessions:
        return "Session not found", 404
    
    session_data = sessions[session_id]
    features = analyzer.extract_real_features(session_data)
    prediction, confidence = analyzer.predict_cognitive_load(features)
    insights = analyzer.get_session_insights(session_data)
    
    # Convert numpy array to dict with proper names
    feature_names = ['keystroke_rate', 'error_freq', 'complexity', 'run_freq', 'edit_ratio', 'syntax_errors', 'runtime_errors']
    features_dict = {name: float(features[i]) for i, name in enumerate(feature_names)}
    
    # Generate feature table rows
    feature_rows = ""
    for name, value in features_dict.items():
        color = "var(--success)" if value < 1 else "var(--warning)" if value < 3 else "var(--error)"
        feature_rows += f'<tr><td>{name.replace("_", " ").title()}</td><td>{value:.2f}</td><td style="color:{color}">●●●</td></tr>'
    
    session_time = time.time() - session_data['start_time']
    total_runs = len(session_data.get('runs', []))
    total_errors = len(session_data.get('errors', []))
    
    pred_class = "pred-low" if prediction == "low" else "pred-medium" if prediction == "medium" else "pred-high"
    
    return f'''
<!DOCTYPE html>
<html>
<head>
<title>Dashboard</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--text:#c9d1d9;--accent:#58a6ff;--success:#238636;--warning:#db6d28;--error:#f85149;}}
body{{font-family:"SF Mono",monospace;background:var(--bg);color:var(--text);padding:40px;max-width:1200px;margin:auto;}}
.card{{background:var(--card);padding:30px;margin:20px 0;border-radius:16px;border:1px solid #30363d;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px;}}
.prediction{{text-align:center;padding:40px;border-radius:20px;font-size:3em;font-weight:700;margin:30px 0;}}
.pred-low{{background:linear-gradient(135deg,var(--success),#166534);color:white;}}
.pred-medium{{background:linear-gradient(135deg,var(--warning),#b65309);color:white;}}
.pred-high{{background:linear-gradient(135deg,var(--error),#da3633);color:white;}}
table{{width:100%;border-collapse:collapse;margin:20px 0;}}th{{background:#21262d;padding:16px;color:var(--accent);text-align:left;}}td{{padding:16px;border-bottom:1px solid #30363d;}}
.btn-back{{background:var(--accent);color:white;padding:14px 28px;border:none;border-radius:8px;cursor:pointer;font-weight:600;font-size:16px;text-decoration:none;display:inline-block;margin-bottom:30px;}}
</style>
</head>
<body>
<a href="/ide/{session_id}" class="btn-back">← Back to IDE</a>
<div class="prediction {pred_class}">
🧠 {prediction.upper()}<br><small>Confidence: {confidence*100:.1f}%</small>
</div>
<div class="grid">
<div class="card">
<h3>📊 Session Metrics</h3>
<table>
<tr><th>Keystrokes</th><td>{insights["total_keystrokes"]}</td></tr>
<tr><th>Code Runs</th><td>{total_runs}</td></tr>
<tr><th>Errors</th><td>{total_errors}</td></tr>
<tr><th>Session Time</th><td>{session_time:.1f}s</td></tr>
<tr><th>Productivity</th><td>{insights["productivity_score"]:.1f}</td></tr>
</table>
</div>
<div class="card">
<h3>🔍 Code Insights</h3>
<table>
<tr><th>Functions Used</th><td>{insights["unique_functions"]}</td></tr>
<tr><th>Avg Lines/Code</th><td>{insights["avg_code_lines"]:.1f}</td></tr>
<tr><th>Top Error</th><td>{insights["most_common_error"][:50]}...</td></tr>
</table>
</div>
</div>
<div class="card">
<h3>📈 Feature Breakdown</h3>
<table>
<tr><th>Feature</th><th>Value</th><th>Impact</th></tr>
{feature_rows}
</table>
</div>
<div class="card" style="grid-column:1/-1;">
<h3>💡 Cognitive Load Explanation</h3>
<p><strong>{prediction.upper()} LOAD</strong> detected because:</p>
<ul style="margin:20px 0;padding-left:25px;">
<li>Keystroke rate: {features_dict["keystroke_rate"]:.1f} chars/min {"(Fast)" if features_dict["keystroke_rate"]>30 else "(Slow)"}</li>
<li>Error rate: {total_errors/max(1,total_runs)*100:.1f}%</li>
<li>Code complexity: {features_dict["complexity"]:.1f} score</li>
<li>{"High edits detected" if features_dict["edit_ratio"]>0.3 else "Stable coding"}</li>
</ul>
<p style="background:#21262d;padding:20px;border-radius:12px;border-left:4px solid var(--accent);">
💡 <strong>Tip:</strong> {"Take a break - high cognitive load detected!" if prediction=="high" else "Good pace! Keep it up." if prediction=="medium" else "Flow state achieved! 🎉"}
</p>
</div>
</body>
</html>'''
if __name__ == '__main__':
    print("🚀 REAL Cognitive Load IDE - Dynamic Analytics!")
    print("📱 http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)