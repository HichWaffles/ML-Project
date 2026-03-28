document.addEventListener('DOMContentLoaded', async () => {
    const wizardStepsContainer = document.getElementById('wizard-steps');
    const wizardHeader = document.getElementById('wizard-header');
    
    const form = document.getElementById('inference-form');
    const submitBtn = document.getElementById('predict-btn');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const resetBtn = document.getElementById('reset-btn');
    const randomSampleBtn = document.getElementById('random-sample-btn');
    const fieldCount = document.getElementById('field-count');
    
    // Result elements
    const placeholder = document.getElementById('result-placeholder');
    const resultCard = document.getElementById('result-card');
    const resultText = document.getElementById('result-text');
    const gaugePath = document.getElementById('gauge-path');
    const riskPercentage = document.getElementById('risk-percentage');
    const predictionLabel = document.getElementById('prediction-label');
    const predictionDetails = document.getElementById('prediction-details');

    const segmentReadout = document.getElementById('segment-readout');
    const segFriction = document.getElementById('seg-friction');
    const segExplorer = document.getElementById('seg-explorer');
    const segTiming = document.getElementById('seg-timing');

    let schema = null;

    const categoryMap = {
        "Demographics": ["AgeCategory", "Gender", "Region", "GeoIP"],
        "Engagement": ["LastLoginIP", "RegistrationDate", "EmailSubscriber", "SatisfactionScore", "CustomerTenureDays"],
        "Transactions": ["MonetaryTotal", "AvgBasketValue", "AvgDaysBetweenPurchases", "PreferredMonth", "PreferredDayOfWeek", "PreferredHour", "WeekendPurchaseRatio"],
        "Behaviors": ["UniqueProducts", "UniqueDescriptions", "AvgProductsPerTransaction", "SpendingCategory", "UniqueCountries"],
        "Support": ["CancelledTransactions", "ReturnRatio", "NegativeQuantityCount", "SupportTicketsCount"]
    };

    let currentStep = 0;
    let totalSteps = 0;

    // 1. Fetch Schema
    try {
        const res = await fetch('/api/schema');
        if (!res.ok) throw new Error("Failed to load schema");
        schema = await res.json();
        
        // 2. Build Wizard Panels dynamically
        const keys = Object.keys(schema);
        fieldCount.textContent = `${keys.length} Features`;
        if (randomSampleBtn) randomSampleBtn.style.display = 'block';
        
        const categorizedKeys = {};
        const usedKeys = new Set();

        for (const [catName, catKeys] of Object.entries(categoryMap)) {
            categorizedKeys[catName] = [];
            for (const k of catKeys) {
                if (schema.hasOwnProperty(k)) {
                    categorizedKeys[catName].push(k);
                    usedKeys.add(k);
                }
            }
        }
        
        const otherKeys = keys.filter(k => !usedKeys.has(k));
        if (otherKeys.length > 0) categorizedKeys["Other"] = otherKeys;

        const categories = Object.keys(categorizedKeys).filter(k => categorizedKeys[k].length > 0);
        totalSteps = categories.length;

        let headerHtml = '';
        
        categories.forEach((catName, idx) => {
            const catKeys = categorizedKeys[catName];
            
            // Header Node
            headerHtml += `
                <div class="wizard-step-node" id="step-node-${idx}">
                    ${idx + 1}
                    <span class="step-label">${catName}</span>
                </div>
            `;
            
            // Panel
            const panel = document.createElement('div');
            panel.className = 'step-panel';
            panel.id = `step-panel-${idx}`;
            
            const title = document.createElement('h3');
            title.textContent = `${idx + 1}. ${catName}`;
            panel.appendChild(title);
            
            const grid = document.createElement('div');
            grid.className = 'form-grid';
            
            catKeys.forEach(key => {
                const field = schema[key];
                const group = document.createElement('div');
                group.className = 'input-group';
                
                const labelText = key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim();
                const label = document.createElement('label');
                label.textContent = labelText;
                label.setAttribute('for', key);
                
                let input;
                if (field.type === 'select') {
                    input = document.createElement('select');
                    field.values.forEach(val => {
                        const opt = document.createElement('option');
                        opt.value = val;
                        opt.textContent = val;
                        if(val === field.default) opt.selected = true;
                        input.appendChild(opt);
                    });
                } else {
                    input = document.createElement('input');
                    input.type = field.type === 'number' ? 'number' : field.type;
                    if (field.type === 'number') input.step = 'any';
                    input.value = field.default !== undefined ? field.default : '';
                    input.placeholder = `Enter ${labelText.toLowerCase()}`;
                }
                
                input.id = key;
                input.name = key;
                input.required = true;
                
                group.appendChild(label);
                group.appendChild(input);
                grid.appendChild(group);
            });
            
            panel.appendChild(grid);
            wizardStepsContainer.appendChild(panel);
        });
        
        wizardHeader.innerHTML = headerHtml;
        updateNav();
        
    } catch (err) {
        fieldCount.textContent = "Error loading form";
        placeholder.innerHTML = `<p style="color:var(--color-risk)">Backend Error: ${err.message}</p>`;
    }

    // Wizard Navigation Logic
    function updateNav() {
        if (totalSteps === 0) return;
        
        prevBtn.classList.toggle('hidden', currentStep === 0);
        
        if (currentStep === totalSteps - 1) {
            nextBtn.classList.add('hidden');
            submitBtn.classList.remove('hidden');
            submitBtn.disabled = false;
        } else {
            nextBtn.classList.remove('hidden');
            submitBtn.classList.add('hidden');
        }
        
        for (let i = 0; i < totalSteps; i++) {
            const panel = document.getElementById(`step-panel-${i}`);
            const node = document.getElementById(`step-node-${i}`);
            
            if (i === currentStep) {
                panel.classList.add('active');
                node.classList.add('active');
                node.classList.remove('completed');
            } else {
                panel.classList.remove('active');
                node.classList.remove('active');
                if (i < currentStep) node.classList.add('completed');
                else node.classList.remove('completed');
            }
        }
    }

    nextBtn.addEventListener('click', () => {
        const currentPanel = document.getElementById(`step-panel-${currentStep}`);
        const inputs = currentPanel.querySelectorAll('input, select');
        let valid = true;
        for (const input of inputs) {
            if (!input.checkValidity()) {
                input.reportValidity();
                valid = false;
                break;
            }
        }
        if (valid && currentStep < totalSteps - 1) {
            currentStep++;
            updateNav();
        }
    });

    prevBtn.addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            updateNav();
        }
    });

    // 3. Handle Submit
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        submitBtn.disabled = true;
        
        // Hide form and show loading in result card area
        wizardStepsContainer.style.display = 'none';
        document.querySelector('.wizard-nav').style.display = 'none';
        wizardHeader.style.display = 'none';
        
        resultCard.classList.add('hidden');
        placeholder.classList.remove('hidden');
        placeholder.classList.add('loading');
        resultText.textContent = "Analyzing risk profile...";

        const payload = {};
        const formData = new FormData(form);
        for (let [key, val] of formData.entries()) {
            const fieldDef = schema[key];
            if (fieldDef.type === 'number') {
                payload[key] = parseFloat(val) || 0;
            } else if (fieldDef.type === 'date') {
                payload[key] = val;
            } else {
                payload[key] = val;
            }
        }

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (!data.success) throw new Error(data.error);
            
            showResult(data.prediction, data.probability, data.segments);
            
        } catch (err) {
            resultText.textContent = `Error: ${err.message}`;
            placeholder.classList.remove('loading');
        } finally {
            submitBtn.disabled = false;
            // Optionally add a "Reset" or "Go Back" button rendering here
        }
    });

    // 4. Update UI with result
    function showResult(pred, prob, segments) {
        placeholder.classList.add('hidden');
        placeholder.classList.remove('loading');
        resultCard.classList.remove('hidden');
        resultCard.classList.remove('low-risk');
        resultCard.classList.remove('high-risk');
        
        const probPct = Math.round(prob * 100);
        
        setTimeout(() => {
            gaugePath.setAttribute('stroke-dasharray', `${probPct}, 100`);
        }, 100);
        
        riskPercentage.textContent = `${probPct}%`;
        
        if (pred === 1) {
            resultCard.classList.add('high-risk');
            predictionLabel.textContent = "High Risk";
            predictionDetails.textContent = "This profile strongly matches patterns of customers who have historically churned. Intervention is highly recommended.";
        } else {
            resultCard.classList.add('low-risk');
            predictionLabel.textContent = "Low Risk";
            predictionDetails.textContent = "This customer profile is currently stable. Based on our model, they are highly likely to remain active.";
        }

        if (segments) {
            segmentReadout.classList.remove('hidden');
            segFriction.textContent = segments['Friction'] || 'Unknown';
            segExplorer.textContent = segments['Explorer'] || 'Unknown';
            segTiming.textContent = segments['Timing'] || 'Unknown';
        } else {
            segmentReadout.classList.add('hidden');
        }
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            currentStep = 0;
            updateNav();
            
            wizardStepsContainer.style.display = 'block';
            document.querySelector('.wizard-nav').style.display = 'flex';
            wizardHeader.style.display = 'flex';
            
            resultCard.classList.add('hidden');
            placeholder.classList.add('hidden');
            
            gaugePath.setAttribute('stroke-dasharray', `0, 100`);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    if (randomSampleBtn) {
        randomSampleBtn.addEventListener('click', async () => {
            const originalText = randomSampleBtn.innerHTML;
            randomSampleBtn.innerHTML = '⏳ Loading...';
            randomSampleBtn.disabled = true;
            
            try {
                const response = await fetch('/api/sample');
                const result = await response.json();
                
                if (result.success && result.data) {
                    // Populate fields
                    for (const [key, value] of Object.entries(result.data)) {
                        const input = document.getElementById(key);
                        if (input) {
                            input.value = value;
                            input.dispatchEvent(new Event('change'));
                        }
                    }
                    // Jump to step 1
                    currentStep = 0;
                    updateNav();
                } else {
                    console.error("Failed to fetch sample:", result.error);
                }
            } catch (err) {
                console.error("Error fetching random sample:", err);
            } finally {
                randomSampleBtn.innerHTML = originalText;
                randomSampleBtn.disabled = false;
            }
        });
    }
});
