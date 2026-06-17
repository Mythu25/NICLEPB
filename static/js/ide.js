let editor, sessionId, sessionStart, runCount = 0, errorCount = 0;
let keystrokeBuffer = [];

function initIDE(id) {
    sessionId = id;
    sessionStart = Date.now();
    
    // Initialize CodeMirror
    editor = CodeMirror.fromTextArea(document.getElementById('codeEditor'), {
        mode: 'python',
        theme: 'dracula',
        lineNumbers: true,
        autofocus: true,
        indentUnit: 4,
        tabSize: 4,
        lineWrapping: true,
        extraKeys: {
            'Ctrl-Enter': runCode,
            'Cmd-Enter': runCode
        }
    });

    // Keystroke logging
    editor.on('change', logKeystroke);
    
    // Update metrics
    updateMetrics();
    setInterval(updateMetrics, 1000);
    
    // Event listeners
    document.getElementById('runBtn').addEventListener('click', runCode);
    document.getElementById('analyzeBtn').addEventListener('click', analyzeLoad);
}

function logKeystroke(cm, change) {
    const timestamp = Date.now() / 1000;
    const key = change.origin === '+input' ? 'input' : 
                change.origin === '+delete' ? 'delete' : 'other';
    
    // Log keystroke
    fetch('/keystroke', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            session_id: sessionId,
            key: key,
            type: change.origin,
            timestamp: timestamp
        })
    });
    
    // Throttle logging
    keystrokeBuffer.push({timestamp, type: key});
}

async function runCode() {
    const code = editor.getValue();
    const status = document.getElementById('status');
    
    status.textContent = 'Executing...';
    status.className = 'status executing';
    
    try {
        const response = await fetch('/execute', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: sessionId, code: code})
        });
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('output').innerHTML = 
                `<div class="output-success">✅ <pre>${result.output || 'No output'}</pre></div>`;
            status.textContent = 'Success';
            status.className = 'status success';
            runCount++;
        } else {
            document.getElementById('output').innerHTML = 
                `<div class="output-error">❌ ${result.error}</div>`;
            status.textContent = 'Error';
            status.className = 'status error';
            errorCount++;
        }
    } catch (error) {
        document.getElementById('output').innerHTML = 
            `<div class="output-error">Network error</div>`;
        status.textContent = 'Error';
        status.className = 'status error';
    }
    
    updateMetrics();
}

async function analyzeLoad() {
    const response = await fetch(`/analyze/${sessionId}`);
    const analysis = await response.json();
    
    // Show quick preview
    alert(`Cognitive Load: ${analysis.prediction} (${Math.round(analysis.confidence * 100)}% confidence)`);
}

function updateMetrics() {
    const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    
    document.getElementById('sessionTime').textContent = 
        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    document.getElementById('runCount').textContent = runCount;
    document.getElementById('errorCount').textContent = errorCount;
}