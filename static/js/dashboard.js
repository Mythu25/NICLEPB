function initDashboard(analysis, modelMetrics) {
    // Load indicator
    const loadLevel = document.querySelector('.load-level');
    loadLevel.className = `load-level level-${analysis.prediction.toLowerCase()}`;
    loadLevel.textContent = analysis.prediction;
    
    // Recommendations
    updateRecommendations(analysis.prediction, analysis.confidence);
    
    // Feature importance chart (mock data for demo)
    createFeatureChart(analysis.features);
    
    // Model metrics
    document.querySelector('.model-metrics').innerHTML += `
        <div>Precision: <strong>${modelMetrics.precision.toFixed(3)}</strong></div>
        <div>Recall: <strong>${modelMetrics.recall.toFixed(3)}</strong></div>
    `;
}

function updateRecommendations(level, confidence) {
    const recs = document.getElementById('recommendations');
    let html = '';
    
    if (level === 'High') {
        html = `
            <div class="recommendation high">
                <h4>🧠 High Cognitive Load Detected</h4>
                <ul>
                    <li>Take a 5-minute break</li>
                    <li>Break problem into smaller parts</li>
                    <li>Review error patterns</li>
                    <li>Simplify your approach</li>
                </ul>
            </div>
        `;
    } else if (level === 'Medium') {
        html = `
            <div class="recommendation medium">
                <h4>⚡ Medium Cognitive Load</h4>
                <ul>
                    <li>Continue at current pace</li>
                    <li>Test incrementally</li>
                    <li>Document your logic</li>
                </ul>
            </div>
        `;
    } else {
        html = `
            <div class="recommendation low">
                <h4>🎉 Low Cognitive Load</h4>
                <ul>
                    <li>Excellent focus!</li>
                    <li>Consider challenging task</li>
                </ul>
            </div>
        `;
    }
    
    recs.innerHTML = html;
}

function createFeatureChart(features) {
    const ctx = document.getElementById('featureChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(features).slice(0, 8),
            datasets: [{
                label: 'Feature Values',
                data: Object.values(features).slice(0, 8),
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}