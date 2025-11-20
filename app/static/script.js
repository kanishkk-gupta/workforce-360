let authState = { sessionId: null, role: null, displayName: null };
let currentSessionId = null;
let hasUploadedCsv = false;
let chartInstances = {};
let employeeFormMode = 'create';
let editingEmployeeId = null;

document.addEventListener('DOMContentLoaded', function() {
    initSliders();

    // Prediction form
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handleFormSubmit);
    }

    // Sidebar toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebarContent = document.getElementById('sidebarContent');
    if (sidebarToggle && sidebarContent) {
        sidebarToggle.addEventListener('click', function() {
            const isVisible = sidebarContent.style.display !== 'none';
            sidebarContent.style.display = isVisible ? 'none' : 'block';
            this.textContent = isVisible ? 'View Unique Values' : 'Hide Unique Values';
        });
    }

    loadUniqueValues();

    // Role radio toggles
    const roleInputs = document.querySelectorAll('input[name="role"]');
    roleInputs.forEach(input => input.addEventListener('change', toggleRoleFields));

    // Login/logout
    const loginForm = document.getElementById('loginForm');
    if (loginForm) loginForm.addEventListener('submit', handleLogin);
    const hrLogoutBtn = document.getElementById('hrLogoutBtn');
    if (hrLogoutBtn) hrLogoutBtn.addEventListener('click', handleLogout);
    const employeeLogoutBtn = document.getElementById('employeeLogoutBtn');
    if (employeeLogoutBtn) employeeLogoutBtn.addEventListener('click', handleLogout);

    // Employee self predict
    const employeePredictBtn = document.getElementById('employeePredictBtn');
    if (employeePredictBtn) employeePredictBtn.addEventListener('click', fetchEmployeePrediction);

    // Feedback
    const feedbackSubmitBtn = document.getElementById('feedbackSubmitBtn');
    if (feedbackSubmitBtn) feedbackSubmitBtn.addEventListener('click', handleFeedbackSubmit);

    // CSV upload inputs & form
    const csvFileInput = document.getElementById('csvFile');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            const fileNameDisplay = document.getElementById('fileName');
            if (fileNameDisplay) {
                fileNameDisplay.textContent = file ? `Selected: ${file.name}` : '';
            }
        });
    }
    const csvUploadForm = document.getElementById('csvUploadForm');
    if (csvUploadForm) csvUploadForm.addEventListener('submit', handleCSVUpload);

    // Employee search
    const employeeSearchForm = document.getElementById('employeeSearchForm');
    if (employeeSearchForm) employeeSearchForm.addEventListener('submit', handleEmployeeSearch);

    // HR Tabs and employee CRUD
    const hrTabButtons = document.getElementById('hrTabButtons');
    if (hrTabButtons) hrTabButtons.addEventListener('click', handleHrTabClick);
    const addEmployeeBtn = document.getElementById('addEmployeeBtn');
    if (addEmployeeBtn) addEmployeeBtn.addEventListener('click', () => openEmployeeForm('create'));
    const employeeForm = document.getElementById('employeeForm');
    if (employeeForm) employeeForm.addEventListener('submit', handleEmployeeFormSubmit);
    const closeFormBtn = document.getElementById('closeEmployeeForm');
    if (closeFormBtn) closeFormBtn.addEventListener('click', closeEmployeeForm);
    const cancelFormBtn = document.getElementById('cancelEmployeeForm');
    if (cancelFormBtn) cancelFormBtn.addEventListener('click', closeEmployeeForm);

    // Refresh dashboard
    const refreshDashboardBtn = document.getElementById('refreshDashboardBtn');
    if (refreshDashboardBtn) refreshDashboardBtn.addEventListener('click', loadAttritionDashboard);

    // Modal close behavior
    const employeeFormModal = document.getElementById('employeeFormModal');
    if (employeeFormModal) {
        employeeFormModal.addEventListener('click', (event) => {
            if (event.target === employeeFormModal) closeEmployeeForm();
        });
    }
    const openUploadModalBtn = document.getElementById('openUploadModalBtn');
    if (openUploadModalBtn) openUploadModalBtn.addEventListener('click', openUploadModal);
    const closeUploadModalBtn = document.getElementById('closeUploadModal');
    if (closeUploadModalBtn) closeUploadModalBtn.addEventListener('click', closeUploadModal);
    const cancelUploadBtn = document.getElementById('cancelUploadBtn');
    if (cancelUploadBtn) cancelUploadBtn.addEventListener('click', closeUploadModal);
    const uploadModal = document.getElementById('uploadModal');
    if (uploadModal) {
        uploadModal.addEventListener('click', (event) => {
            if (event.target === uploadModal) closeUploadModal();
        });
    }

    // Results container init
    const resultsContainer = document.getElementById('resultsContainer');
    if (resultsContainer && !resultsContainer.dataset.hasResult) {
        resultsContainer.dataset.hasResult = 'false';
    }

    mountPredictionWorkspace('employeePredictionHost');
    syncResultsVisibility();

    toggleRoleFields();
    updatePortalVisibility();
});

function initSliders() {
    const sliders = document.querySelectorAll('.slider');
    sliders.forEach(slider => {
        const valueDisplay = document.getElementById(slider.id + '-value');
        if (valueDisplay) {
            slider.addEventListener('input', function() {
                if (this.id === 'monthly_income') {
                    valueDisplay.textContent = parseInt(this.value, 10).toLocaleString();
                } else {
                    valueDisplay.textContent = this.value;
                }
            });
            // initialize display
            valueDisplay.textContent = slider.id === 'monthly_income'
                ? parseInt(slider.value || 0, 10).toLocaleString()
                : slider.value;
        }
    });

    // relationship_satisfaction explicit handling (if present)
    const relationshipSatisfactionSlider = document.getElementById('relationship_satisfaction');
    const relationshipSatisfactionDisplay = document.getElementById('relationship_satisfaction-value');
    if (relationshipSatisfactionSlider && relationshipSatisfactionDisplay) {
        relationshipSatisfactionSlider.addEventListener('input', function() {
            relationshipSatisfactionDisplay.textContent = this.value;
        });
        relationshipSatisfactionDisplay.textContent = relationshipSatisfactionSlider.value;
    }

    // monthly_income formatting (already covered but keep safe)
    const monthlyIncomeSlider = document.getElementById('monthly_income');
    const monthlyIncomeDisplay = document.getElementById('monthly_income-value');
    if (monthlyIncomeSlider && monthlyIncomeDisplay) {
        monthlyIncomeSlider.addEventListener('input', function() {
            monthlyIncomeDisplay.textContent = parseInt(this.value || 0, 10).toLocaleString();
        });
        monthlyIncomeDisplay.textContent = parseInt(monthlyIncomeSlider.value || 0, 10).toLocaleString();
    }
}

async function handleFormSubmit(e) {
    e.preventDefault();

    const submitButton = document.querySelector('.predict-btn');
    const originalText = submitButton ? submitButton.textContent : 'Predict';

    if (submitButton) {
        submitButton.innerHTML = '<span class="loading"></span> Predicting...';
        submitButton.disabled = true;
    }

    try {
        const formData = {
            age: parseInt(document.getElementById('age').value, 10),
            department: document.getElementById('department').value,
            environment_satisfaction: parseInt(document.getElementById('environment_satisfaction').value, 10),
            job_role: document.getElementById('job_role').value,
            job_satisfaction: parseInt(document.getElementById('job_satisfaction').value, 10),
            monthly_income: parseInt(document.getElementById('monthly_income').value, 10),
            num_companies_worked: parseInt(document.getElementById('num_companies_worked').value, 10),
            over_time: document.getElementById('over_time').checked,
            percent_salary_hike: parseInt(document.getElementById('percent_salary_hike').value, 10),
            relationship_satisfaction: parseInt(document.getElementById('relationship_satisfaction').value, 10),
            training_times_last_year: parseInt(document.getElementById('training_times_last_year').value, 10),
            work_life_balance: parseInt(document.getElementById('work_life_balance').value, 10),
            years_since_last_promotion: parseInt(document.getElementById('years_since_last_promotion').value, 10),
            years_with_curr_manager: parseInt(document.getElementById('years_with_curr_manager').value, 10)
        };

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...buildAuthHeaders() },
            body: JSON.stringify(formData)
        });

        if (!response.ok) throw new Error('Prediction failed');

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while making the prediction. Please try again.');
    } finally {
        if (submitButton) {
            submitButton.textContent = originalText;
            submitButton.disabled = false;
        }
    }
}

function displayResults(result) {
    const resultsContainer = document.getElementById('resultsContainer');
    const predictionResult = document.getElementById('predictionResult');
    const probabilityFill = document.getElementById('probabilityFill');
    const probabilityValue = document.getElementById('probabilityValue');
    const explanationSection = document.getElementById('explanationSection');
    const topFeaturesList = document.getElementById('topFeaturesList');
    const recommendationsList = document.getElementById('recommendationsList');
    const recommendationsHeader = document.getElementById('recommendationsHeader');

    if (resultsContainer) {
        resultsContainer.dataset.hasResult = 'true';
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    const probability = result.probability ?? 0;
    const percentage = (probability * 100).toFixed(2);
    const threshold = result.threshold_used ?? 0.45;
    const thresholdPercentage = (threshold * 100).toFixed(0);
    const riskLevel = result.risk_level ?? (result.will_leave ? 'HIGH RISK - Will Likely Leave' : 'LOW RISK - Will Likely Stay');

    if (predictionResult) {
        if (result.will_leave) {
            predictionResult.innerHTML = `
                <div style="margin-bottom: 10px;">
                    <strong style="font-size: 1.1em;">⚠️ Employee is predicted to LEAVE (Attrition = Yes)</strong>
                </div>
                <div style="font-size: 0.95em; color: #555; margin-top: 8px;">
                    <div><strong>Risk Level:</strong> ${riskLevel}</div>
                    <div><strong>Attrition Probability:</strong> ${percentage}%</div>
                    <div style="color: #d9534f;"><strong>Decision Threshold:</strong> ≥ ${thresholdPercentage}% (employee probability exceeds threshold)</div>
                </div>
            `;
            predictionResult.className = 'prediction-result leave';
        } else {
            predictionResult.innerHTML = `
                <div style="margin-bottom: 10px;">
                    <strong style="font-size: 1.1em;">✅ Employee is predicted to STAY (Attrition = No)</strong>
                </div>
                <div style="font-size: 0.95em; color: #555; margin-top: 8px;">
                    <div><strong>Risk Level:</strong> ${riskLevel}</div>
                    <div><strong>Attrition Probability:</strong> ${percentage}%</div>
                    <div style="color: #27AE60;"><strong>Decision Threshold:</strong> ≥ ${thresholdPercentage}% (employee probability is below threshold)</div>
                </div>
            `;
            predictionResult.className = 'prediction-result stay';
        }
    }

    if (probabilityFill) probabilityFill.style.width = percentage + '%';
    if (probabilityValue) probabilityValue.textContent = percentage + '%';

    if (probability < 0.3) {
        if (probabilityFill) probabilityFill.style.background = 'linear-gradient(90deg, #27AE60, #229954)';
    } else if (probability < 0.6) {
        if (probabilityFill) probabilityFill.style.background = 'linear-gradient(90deg, #F39C12, #E67E22)';
    } else {
        if (probabilityFill) probabilityFill.style.background = 'linear-gradient(90deg, #E74C3C, #C0392B)';
    }

    // Recommendations visibility (HR only)
    const showRecommendations = authState.role === 'hr';
    if (recommendationsHeader) recommendationsHeader.style.display = showRecommendations ? 'block' : 'none';
    if (recommendationsList) recommendationsList.style.display = showRecommendations ? 'block' : 'none';

    if (topFeaturesList) topFeaturesList.innerHTML = '';
    if (recommendationsList && showRecommendations) recommendationsList.innerHTML = '';

    if (result.explanation) {
        if (explanationSection) explanationSection.style.display = 'block';
        const explanation = result.explanation;

        if (Array.isArray(explanation.top_features) && explanation.top_features.length && topFeaturesList) {
            explanation.top_features.forEach(tf => {
                const item = document.createElement('div');
                item.className = 'top-feature-item';
                const name = document.createElement('strong');
                name.textContent = tf.feature + ': ';
                const detail = document.createElement('span');
                const valText = tf.value === null || tf.value === undefined ? '' : `value = ${tf.value}`;
                const shapText = ` (contribution = ${tf.shap_value.toFixed(3)}, importance = ${(tf.relative_importance*100).toFixed(1)}%)`;
                detail.textContent = valText + shapText;
                item.appendChild(name);
                item.appendChild(detail);
                topFeaturesList.appendChild(item);
            });
        }

        if (Array.isArray(explanation.recommendations) && explanation.recommendations.length && recommendationsList && showRecommendations) {
            explanation.recommendations.forEach(rec => {
                const card = document.createElement('div');
                card.className = 'recommendation-card';
                const title = document.createElement('strong');
                title.textContent = rec.feature ? `${rec.feature}: ` : '';
                const text = document.createElement('div');
                text.textContent = rec.issue;
                const severity = document.createElement('div');
                severity.className = 'rec-severity';
                severity.textContent = rec.severity ? `Severity: ${(rec.severity*100).toFixed(1)}%` : '';
                card.appendChild(title);
                card.appendChild(text);
                card.appendChild(severity);
                recommendationsList.appendChild(card);
            });
        }
    } else if (result.explanation_error) {
        if (explanationSection) explanationSection.style.display = 'block';
        if (recommendationsList) {
            const err = document.createElement('div');
            err.className = 'explanation-error';
            err.textContent = 'Explanation not available: ' + result.explanation_error;
            recommendationsList.appendChild(err);
        }
    } else {
        if (explanationSection) explanationSection.style.display = 'none';
    }

    syncResultsVisibility();
}

async function loadUniqueValues() {
    try {
        const response = await fetch('/api/unique_values');
        if (!response.ok) throw new Error('Failed to load unique values');
        const uniqueValues = await response.json();
        const uniqueValuesList = document.getElementById('uniqueValuesList');
        if (!uniqueValuesList) return;
        uniqueValuesList.innerHTML = '';
        for (const [column, values] of Object.entries(uniqueValues)) {
            const item = document.createElement('div');
            item.className = 'unique-value-item';
            const strong = document.createElement('strong');
            strong.textContent = column + ':';
            const span = document.createElement('span');
            if (Array.isArray(values)) {
                const formattedValues = values.slice(0, 10).map(v => typeof v === 'boolean' ? v.toString() : v).join(', ');
                const more = values.length > 10 ? ` ... (+${values.length - 10} more)` : '';
                span.textContent = formattedValues + more;
            } else {
                span.textContent = values.toString();
            }
            item.appendChild(strong);
            item.appendChild(span);
            uniqueValuesList.appendChild(item);
        }
    } catch (error) {
        console.error('Error loading unique values:', error);
    }
}

async function handleCSVUpload(e) {
    e.preventDefault();

    const fileInput = document.getElementById('csvFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');

    if (authState.role !== 'hr') {
        if (uploadStatus) {
            uploadStatus.textContent = 'Only HR users can upload employee datasets.';
            uploadStatus.className = 'upload-status error';
        }
        return;
    }

    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        if (uploadStatus) {
            uploadStatus.textContent = 'Please select a CSV file first.';
            uploadStatus.className = 'upload-status error';
        }
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const originalText = uploadBtn ? uploadBtn.textContent : 'Upload';
    if (uploadBtn) {
        uploadBtn.innerHTML = '<span class="loading"></span> Uploading...';
        uploadBtn.disabled = true;
    }
    if (uploadStatus) {
        uploadStatus.className = 'upload-status';
        uploadStatus.textContent = '';
    }

    try {
        const response = await fetch('/api/upload_csv', {
            method: 'POST',
            headers: buildAuthHeaders(), // session header will be included if available
            body: formData
        });

        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Upload failed');

        hasUploadedCsv = true;
        currentSessionId = result.session_id || authState.sessionId || currentSessionId;
        if (authState.sessionId == null && result.session_id) {
            authState.sessionId = result.session_id;
        }

        if (uploadStatus) {
            uploadStatus.textContent = `✅ ${result.message} - ${result.rows} rows loaded. Available columns: ${result.columns.length}`;
            uploadStatus.className = 'upload-status success';
        }

        closeUploadModal();
        const hrTabsWrapper = document.getElementById('hrTabsWrapper');
        if (hrTabsWrapper) hrTabsWrapper.style.display = 'flex';
        setHrActiveTab('hrAttritionTab');
        mountPredictionWorkspace('hrPredictionHost');

        const searchSection = document.getElementById('searchSection');
        if (searchSection) {
            searchSection.style.display = 'block';
            searchSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        loadEmployeeCards();
        loadAttritionDashboard();
        loadFeedbackSummary();
        syncResultsVisibility();
    } catch (error) {
        console.error('Error uploading CSV:', error);
        if (uploadStatus) {
            uploadStatus.textContent = `❌ Error: ${error.message}`;
            uploadStatus.className = 'upload-status error';
        }
    } finally {
        if (uploadBtn) {
            uploadBtn.textContent = originalText;
            uploadBtn.disabled = false;
        }
    }
}

async function handleEmployeeSearch(e) {
    e.preventDefault();

    const employeeNameInput = document.getElementById('employeeName');
    const searchBtn = document.getElementById('searchBtn');
    const searchStatus = document.getElementById('searchStatus');
    const employeeName = employeeNameInput ? employeeNameInput.value.trim() : '';

    if (!employeeName) {
        if (searchStatus) {
            searchStatus.textContent = 'Please enter an employee name or ID.';
            searchStatus.className = 'search-status error';
        }
        return;
    }

    if (authState.role !== 'hr') {
        if (searchStatus) {
            searchStatus.textContent = 'Please sign in as HR to search employees.';
            searchStatus.className = 'search-status error';
        }
        return;
    }

    if (!hasUploadedCsv && !currentSessionId) {
        if (searchStatus) {
            searchStatus.textContent = 'Please upload a CSV file first.';
            searchStatus.className = 'search-status error';
        }
        return;
    }

    const originalText = searchBtn ? searchBtn.textContent : 'Search';
    if (searchBtn) {
        searchBtn.innerHTML = '<span class="loading"></span> Searching...';
        searchBtn.disabled = true;
    }
    if (searchStatus) {
        searchStatus.className = 'search-status';
        searchStatus.textContent = '';
    }

    try {
        const response = await fetch('/api/search_employee', {
            method: 'POST',
            headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ employee_name: employeeName })
        });

        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Search failed');

        if (searchStatus) {
            searchStatus.textContent = `✅ Found employee: ${result.employee_name || employeeName}`;
            searchStatus.className = 'search-status success';
        }

        if (result.employee_data) populateFormWithEmployeeData(result.employee_data);
        displayResults(result);
    } catch (error) {
        console.error('Error searching employee:', error);
        if (searchStatus) {
            searchStatus.textContent = `❌ Error: ${error.message}`;
            searchStatus.className = 'search-status error';
        }
    } finally {
        if (searchBtn) {
            searchBtn.textContent = originalText;
            searchBtn.disabled = false;
        }
    }
}

function populateFormWithEmployeeData(employeeData) {
    if (!employeeData || typeof employeeData !== 'object') return;

    const sliderIds = {
        age: 'Age',
        environment_satisfaction: 'EnvironmentSatisfaction',
        job_satisfaction: 'JobSatisfaction',
        monthly_income: 'MonthlyIncome',
        num_companies_worked: 'NumCompaniesWorked',
        percent_salary_hike: 'PercentSalaryHike',
        relationship_satisfaction: 'RelationshipSatisfaction',
        training_times_last_year: 'TrainingTimesLastYear',
        work_life_balance: 'WorkLifeBalance',
        years_since_last_promotion: 'YearsSinceLastPromotion',
        years_with_curr_manager: 'YearsWithCurrManager'
    };

    Object.entries(sliderIds).forEach(([inputId, dataKey]) => {
        const value = employeeData[dataKey];
        if (value !== undefined && value !== null) {
            const slider = document.getElementById(inputId);
            if (slider) {
                slider.value = Number(value);
                slider.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    });

    if (employeeData.Department) {
        const departmentSelect = document.getElementById('department');
        if (departmentSelect) {
            let option = Array.from(departmentSelect.options).find(opt => opt.value === employeeData.Department);
            if (!option) {
                option = new Option(employeeData.Department, employeeData.Department);
                departmentSelect.appendChild(option);
            }
            departmentSelect.value = employeeData.Department;
        }
    }

    if (employeeData.JobRole) {
        const jobRoleSelect = document.getElementById('job_role');
        if (jobRoleSelect) {
            let option = Array.from(jobRoleSelect.options).find(opt => opt.value === employeeData.JobRole);
            if (!option) {
                option = new Option(employeeData.JobRole, employeeData.JobRole);
                jobRoleSelect.appendChild(option);
            }
            jobRoleSelect.value = employeeData.JobRole;
        }
    }

    if (employeeData.OverTime !== undefined) {
        const overTimeCheckbox = document.getElementById('over_time');
        if (overTimeCheckbox) {
            if (typeof employeeData.OverTime === 'boolean') {
                overTimeCheckbox.checked = employeeData.OverTime;
            } else if (typeof employeeData.OverTime === 'string') {
                overTimeCheckbox.checked = ['TRUE', 'YES', 'Y', '1'].includes(employeeData.OverTime.toUpperCase());
            } else {
                overTimeCheckbox.checked = Boolean(employeeData.OverTime);
            }
        }
    }
}

function handleHrTabClick(event) {
    const button = event.target.closest('.tab-btn');
    if (!button) return;
    setHrActiveTab(button.dataset.tab);
}

let activeHrTab = 'hrAttritionTab';
function setHrActiveTab(tabId) {
    activeHrTab = tabId;
    const tabButtons = document.querySelectorAll('#hrTabButtons .tab-btn');
    tabButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabId));
    const panels = document.querySelectorAll('#hrTabsWrapper .tab-panel');
    panels.forEach(panel => panel.classList.toggle('active', panel.id === tabId));
    if (tabId === 'hrPredictTab') mountPredictionWorkspace('hrPredictionHost');
    syncResultsVisibility();
}

function openUploadModal() {
    const modal = document.getElementById('uploadModal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('show');
    }
}

function closeUploadModal() {
    const modal = document.getElementById('uploadModal');
    if (modal) {
        modal.classList.remove('show');
        modal.classList.add('hidden');
    }
    const uploadStatus = document.getElementById('uploadStatus');
    if (uploadStatus) {
        uploadStatus.textContent = '';
        uploadStatus.className = 'upload-status';
    }
    const fileInput = document.getElementById('csvFile');
    if (fileInput) fileInput.value = '';
    const fileName = document.getElementById('fileName');
    if (fileName) fileName.textContent = '';
}

function mountPredictionWorkspace(hostId) {
    const host = document.getElementById(hostId);
    const workspace = document.getElementById('predictionWorkspace');
    if (host && workspace && host !== workspace.parentElement) {
        host.appendChild(workspace);
    }
}

function syncResultsVisibility() {
    const resultsContainer = document.getElementById('resultsContainer');
    if (!resultsContainer) return;
    const hasResult = resultsContainer.dataset.hasResult === 'true';
    if (!hasResult) {
        resultsContainer.style.display = 'none';
        return;
    }
    if (authState.role === 'hr') {
        const predictPanel = document.getElementById('hrPredictTab');
        if (predictPanel && predictPanel.classList.contains('active')) {
            resultsContainer.style.display = 'block';
        } else {
            resultsContainer.style.display = 'none';
        }
    } else if (authState.role === 'employee') {
        resultsContainer.style.display = 'block';
    } else {
        resultsContainer.style.display = 'none';
    }
}

function openEmployeeForm(mode, employee = {}) {
    employeeFormMode = mode;
    editingEmployeeId = mode === 'edit' ? employee.employee_number : null;
    const title = document.getElementById('employeeFormTitle');
    const modal = document.getElementById('employeeFormModal');
    if (title) title.textContent = mode === 'edit' ? 'Edit Employee' : 'Add Employee';
    document.getElementById('employeeNumberField').value = employee.employee_number || '';
    document.getElementById('employeeNameField').value = employee.name || '';
    document.getElementById('employeeDepartmentField').value = employee.department || '';
    document.getElementById('employeeRoleField').value = employee.job_role || '';
    document.getElementById('employeeAgeField').value = employee.age || 30;
    document.getElementById('employeeIncomeField').value = employee.monthly_income || 5000;
    document.getElementById('employeeAttritionField').value = employee.attrition ? 'true' : 'false';
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('show');
    }
}

function closeEmployeeForm() {
    const modal = document.getElementById('employeeFormModal');
    if (modal) {
        modal.classList.remove('show');
        modal.classList.add('hidden');
    }
    editingEmployeeId = null;
}

async function handleEmployeeFormSubmit(e) {
    e.preventDefault();
    if (authState.role !== 'hr') return;
    const payload = {
        name: document.getElementById('employeeNameField').value.trim(),
        department: document.getElementById('employeeDepartmentField').value.trim(),
        job_role: document.getElementById('employeeRoleField').value.trim(),
        age: parseInt(document.getElementById('employeeAgeField').value, 10),
        monthly_income: parseFloat(document.getElementById('employeeIncomeField').value),
        attrition: document.getElementById('employeeAttritionField').value === 'true'
    };
    if (!payload.name) { alert('Please provide a name.'); return; }
    if (isNaN(payload.age) || isNaN(payload.monthly_income)) { alert('Please provide valid numeric values for age and monthly income.'); return; }

    const isEdit = employeeFormMode === 'edit' && editingEmployeeId;
    const url = isEdit ? `/api/hr/employees/${editingEmployeeId}` : '/api/hr/employees';
    const method = isEdit ? 'PUT' : 'POST';

    try {
        const response = await fetch(url, {
            method,
            headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Unable to save employee.');
        closeEmployeeForm();
        loadEmployeeCards();
        loadAttritionDashboard();
        alert(result.message);
    } catch (error) {
        console.error('Employee save error:', error);
        alert(error.message);
    }
}

async function loadEmployeeCards() {
    const cardsContainer = document.getElementById('employeeCards');
    if (!cardsContainer || authState.role !== 'hr') return;
    if (!hasUploadedCsv) {
        cardsContainer.innerHTML = '<p class="helper-text">Upload a CSV to see employees listed here.</p>';
        return;
    }
    cardsContainer.innerHTML = '<p class="helper-text"><span class="loading"></span> Loading employees...</p>';
    try {
        const response = await fetch('/api/hr/employees', { method: 'GET', headers: buildAuthHeaders() });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Unable to load employees.');
        renderEmployeeCards(data.employees || []);
    } catch (error) {
        console.error('Employee load error:', error);
        cardsContainer.innerHTML = `<p class="helper-text">${error.message}</p>`;
    }
}

function renderEmployeeCards(employees) {
    const container = document.getElementById('employeeCards');
    if (!container) return;
    container.innerHTML = '';
    if (!employees.length) {
        container.innerHTML = '<p class="helper-text">No employees found. Upload a dataset or add a new record.</p>';
        return;
    }
    employees.forEach(employee => {
        const card = document.createElement('div');
        card.className = 'employee-card';

        const header = document.createElement('div');
        header.className = 'card-header';
        const title = document.createElement('h3');
        title.textContent = employee.name || `Employee ${employee.employee_number}`;
        const badge = document.createElement('span');
        badge.className = `badge ${employee.attrition ? 'warning' : 'good'}`;
        badge.textContent = employee.attrition ? 'Attrition Risk' : 'Stable';
        header.appendChild(title);
        header.appendChild(badge);

        const infoList = document.createElement('div');
        infoList.innerHTML = `
            <p><strong>ID:</strong> ${employee.employee_number || '-'}</p>
            <p><strong>Department:</strong> ${employee.department || '-'}</p>
            <p><strong>Role:</strong> ${employee.job_role || '-'}</p>
            <p><strong>Age:</strong> ${employee.age || '-'}</p>
            <p><strong>Monthly Income:</strong> ${formatCurrency(employee.monthly_income)}</p>
        `;

        const actions = document.createElement('div');
        actions.className = 'card-actions';
        const editBtn = document.createElement('button');
        editBtn.className = 'edit-btn';
        editBtn.textContent = 'Edit';
        editBtn.addEventListener('click', () => openEmployeeForm('edit', employee));
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.textContent = 'Delete';
        deleteBtn.addEventListener('click', () => deleteEmployee(employee.employee_number));
        actions.appendChild(editBtn);
        actions.appendChild(deleteBtn);

        card.appendChild(header);
        card.appendChild(infoList);
        card.appendChild(actions);
        container.appendChild(card);
    });
}

async function deleteEmployee(employeeNumber) {
    if (!employeeNumber) return;
    const confirmed = confirm(`Delete employee #${employeeNumber}?`);
    if (!confirmed) return;
    try {
        const response = await fetch(`/api/hr/employees/${employeeNumber}`, {
            method: 'DELETE',
            headers: buildAuthHeaders()
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Unable to delete employee.');
        loadEmployeeCards();
        loadAttritionDashboard();
        alert(result.message);
    } catch (error) {
        console.error('Delete error:', error);
        alert(error.message);
    }
}

async function loadAttritionDashboard() {
    const statusEl = document.getElementById('attritionDashboardStatus');
    if (authState.role !== 'hr') return;
    if (!statusEl) return;
    if (!hasUploadedCsv) {
        statusEl.textContent = 'Upload a CSV to generate the attrition dashboard.';
        statusEl.className = 'upload-status';
        return;
    }
    statusEl.innerHTML = '<span class="loading"></span> Building dashboard...';
    statusEl.className = 'upload-status';
    try {
        const response = await fetch('/api/hr/attrition_dashboard', { method: 'GET', headers: buildAuthHeaders() });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Unable to load dashboard.');
        renderAttritionDashboard(data);
        statusEl.textContent = '';
        statusEl.className = 'upload-status';
    } catch (error) {
        console.error('Attrition dashboard error:', error);
        statusEl.textContent = `❌ ${error.message}`;
        statusEl.className = 'upload-status error';
    }
}

function renderAttritionDashboard(data) {
    const kpiContainer = document.getElementById('attritionKpis');
    if (kpiContainer) {
        kpiContainer.innerHTML = '';
        const kpis = data.kpis || {};
        const cards = [
            { label: 'Total Employees', value: kpis.total_employees },
            { label: 'Total Attrition', value: kpis.total_attrition },
            { label: 'Attrition Rate', value: `${kpis.attrition_rate || 0}%` },
            { label: 'Average Age', value: kpis.average_age || '-' },
            { label: 'Avg Monthly Income', value: formatCurrency(kpis.average_income) }
        ];
        cards.forEach(card => {
            const node = document.createElement('div');
            node.className = 'kpi-card';
            node.innerHTML = `<span>${card.label}</span><strong>${card.value ?? '-'}</strong>`;
            kpiContainer.appendChild(node);
        });
    }

    renderChart('attritionDistributionChart', 'doughnut', data.distribution?.labels || [], data.distribution?.values || []);
    renderChart('jobLevelChart', 'bar', data.job_level?.labels || [], data.job_level?.values || []);
    renderChart('ageGroupChart', 'bar', data.age_group?.labels || [], data.age_group?.values || []);
    renderChart('trainingChart', 'bar', data.training?.labels || [], data.training?.values || []);
    renderChart('environmentChart', 'doughnut', data.environment?.labels || [], data.environment?.values || []);
    renderChart('jobSatisfactionChart', 'bar', data.job_satisfaction?.labels || [], data.job_satisfaction?.values || []);
    renderChart('promotionChart', 'bar', data.promotion?.labels || [], data.promotion?.values || []);
    renderChart('workLifeChart', 'doughnut', data.work_life?.labels || [], data.work_life?.values || []);

    const recommendationsList = document.getElementById('attritionRecommendations');
    if (recommendationsList) {
        recommendationsList.innerHTML = '';
        const recs = data.recommendations || [];
        if (recs.length) {
            recs.forEach(rec => {
                const item = document.createElement('div');
                item.className = 'recommendation-item';
                const title = document.createElement('h4');
                title.textContent = rec.title || rec.action || 'Recommendation';
                const detail = document.createElement('p');
                detail.textContent = rec.detail || rec.action || 'Act on this insight.';
                item.appendChild(title);
                item.appendChild(detail);
                recommendationsList.appendChild(item);
            });
        } else {
            recommendationsList.innerHTML = '<p class="helper-text">Gemini recommendations will appear here.</p>';
        }
    }
}

function renderChart(canvasId, type, labels = [], values = []) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;
    if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
    const palette = ['#2E86DE', '#48C9B0', '#F5B041', '#D98880', '#7D3C98', '#16A085', '#F4D03F', '#BB8FCE'];
    chartInstances[canvasId] = new Chart(canvas, {
        type,
        data: {
            labels,
            datasets: [{
                label: 'Employees',
                data: values,
                backgroundColor: type === 'doughnut' ? palette.slice(0, labels.length || 1) : '#4A90E2',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: type === 'doughnut' },
                tooltip: { enabled: true }
            },
            cutout: type === 'doughnut' ? '65%' : undefined,
            scales: type === 'bar' ? {
                y: { beginAtZero: true, ticks: { precision: 0 } },
                x: { ticks: { autoSkip: false } }
            } : {}
        }
    });
}

function formatCurrency(value) {
    if (value === null || value === undefined || isNaN(value)) return '-';
    return '$' + Number(value).toLocaleString();
}

function toggleRoleFields() {
    const selectedRole = document.querySelector('input[name="role"]:checked');
    const hrFields = document.getElementById('hrFields');
    const employeeFields = document.getElementById('employeeFields');
    if (!selectedRole || !hrFields || !employeeFields) return;
    if (selectedRole.value === 'hr') {
        hrFields.style.display = 'block';
        employeeFields.style.display = 'none';
    } else {
        hrFields.style.display = 'none';
        employeeFields.style.display = 'block';
    }
}

function updatePortalVisibility() {
    const loginSection = document.getElementById('loginSection');
    const hrPortal = document.getElementById('hrPortal');
    const employeePortal = document.getElementById('employeePortal');
    const sidebar = document.getElementById('sidebar');

    if (authState.role === 'hr') {
        if (loginSection) loginSection.style.display = 'none';
        if (hrPortal) hrPortal.style.display = 'block';
        if (employeePortal) employeePortal.style.display = 'none';
        if (sidebar) sidebar.style.display = 'block';
        mountPredictionWorkspace('hrPredictionHost');
    } else if (authState.role === 'employee') {
        if (loginSection) loginSection.style.display = 'none';
        if (hrPortal) hrPortal.style.display = 'none';
        if (employeePortal) employeePortal.style.display = 'block';
        if (sidebar) sidebar.style.display = 'none';
        mountPredictionWorkspace('employeePredictionHost');
    } else {
        if (loginSection) loginSection.style.display = 'block';
        if (hrPortal) hrPortal.style.display = 'none';
        if (employeePortal) employeePortal.style.display = 'none';
        if (sidebar) sidebar.style.display = 'none';
        mountPredictionWorkspace('employeePredictionHost');
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) resultsContainer.style.display = 'none';
    }
    syncResultsVisibility();
}

function buildAuthHeaders(base = {}) {
    const headers = { ...base };
    const sessionHeader = authState.sessionId || currentSessionId;
    if (sessionHeader) headers['X-Session-ID'] = sessionHeader;
    return headers;
}

async function handleLogin(e) {
    e.preventDefault();
    const roleInput = document.querySelector('input[name="role"]:checked');
    const loginStatus = document.getElementById('loginStatus');
    if (!roleInput) return;
    const role = roleInput.value;
    const payload = { role };

    if (role === 'hr') {
        payload.username = document.getElementById('hrUsername').value.trim();
        payload.password = document.getElementById('hrPassword').value;
        if (!payload.username || !payload.password) {
            if (loginStatus) { loginStatus.textContent = 'Enter HR username and password.'; loginStatus.className = 'upload-status error'; }
            return;
        }
    } else {
        payload.employee_id = document.getElementById('employeeIdLogin').value.trim();
        if (!payload.employee_id) {
            if (loginStatus) { loginStatus.textContent = 'Please provide your Employee ID.'; loginStatus.className = 'upload-status error'; }
            return;
        }
    }

    const loginBtn = document.getElementById('loginBtn');
    const originalText = loginBtn ? loginBtn.textContent : 'Sign in';
    if (loginBtn) { loginBtn.innerHTML = '<span class="loading"></span> Signing in...'; loginBtn.disabled = true; }
    if (loginStatus) { loginStatus.textContent = ''; loginStatus.className = 'upload-status'; }

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Unable to sign in.');

        authState = {
            sessionId: result.session_id,
            role: result.role,
            displayName: result.display_name || ''
        };
        currentSessionId = result.session_id;
        hasUploadedCsv = false;
        const searchSection = document.getElementById('searchSection');
        if (searchSection) searchSection.style.display = 'none';

        if (loginStatus) { loginStatus.textContent = `Welcome ${result.display_name || ''}!`; loginStatus.className = 'upload-status success'; }

        updatePortalVisibility();
        if (authState.role === 'hr') {
            const hrTabsWrapper = document.getElementById('hrTabsWrapper');
            if (hrTabsWrapper) hrTabsWrapper.style.display = hasUploadedCsv ? 'flex' : 'none';
            mountPredictionWorkspace('hrPredictionHost');
        } else {
            mountPredictionWorkspace('employeePredictionHost');
        }
        syncResultsVisibility();

        if (authState.role === 'employee') {
            const welcome = document.getElementById('employeeWelcome');
            if (welcome) welcome.textContent = `Signed in as ${result.display_name}`;
            fetchEmployeePrediction();
        } else if (authState.role === 'hr') {
            loadFeedbackSummary();
        }
    } catch (error) {
        console.error('Login error:', error);
        if (loginStatus) { loginStatus.textContent = `❌ ${error.message}`; loginStatus.className = 'upload-status error'; }
    } finally {
        if (loginBtn) { loginBtn.textContent = originalText; loginBtn.disabled = false; }
    }
}

async function handleLogout() {
    try {
        if (authState.sessionId) {
            await fetch('/api/logout', { method: 'POST', headers: buildAuthHeaders() });
        }
    } catch (error) {
        console.warn('Logout error:', error);
    }

    authState = { sessionId: null, role: null, displayName: null };
    currentSessionId = null;
    hasUploadedCsv = false;
    const resultsContainer = document.getElementById('resultsContainer');
    if (resultsContainer) { resultsContainer.dataset.hasResult = 'false'; resultsContainer.style.display = 'none'; }
    const searchSection = document.getElementById('searchSection');
    if (searchSection) searchSection.style.display = 'none';
    const loginStatus = document.getElementById('loginStatus');
    if (loginStatus) { loginStatus.textContent = 'You have been signed out.'; loginStatus.className = 'upload-status'; }
    closeEmployeeForm();
    closeUploadModal();
    const cards = document.getElementById('employeeCards');
    if (cards) cards.innerHTML = '<p class="helper-text">Upload a CSV to see employees listed here.</p>';
    const dashboardStatus = document.getElementById('attritionDashboardStatus');
    if (dashboardStatus) { dashboardStatus.textContent = ''; dashboardStatus.className = 'upload-status'; }
    const hrTabsWrapper = document.getElementById('hrTabsWrapper');
    if (hrTabsWrapper) hrTabsWrapper.style.display = 'none';
    setHrActiveTab('hrAttritionTab');
    mountPredictionWorkspace('employeePredictionHost');
    syncResultsVisibility();
    updatePortalVisibility();
}

async function fetchEmployeePrediction() {
    if (authState.role !== 'employee') return;
    const statusEl = document.getElementById('employeeSelfStatus');
    if (statusEl) { statusEl.innerHTML = '<span class="loading"></span> Fetching your prediction...'; statusEl.className = 'upload-status'; statusEl.style.display = 'block'; }
    try {
        const response = await fetch('/api/employee_self', { method: 'GET', headers: buildAuthHeaders() });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Unable to fetch prediction.');
        if (statusEl) { statusEl.textContent = 'Prediction refreshed from the latest data.'; statusEl.className = 'upload-status success'; }
        displayResults(result);
    } catch (error) {
        console.error('Employee prediction error:', error);
        if (statusEl) { statusEl.textContent = `❌ ${error.message}`; statusEl.className = 'upload-status error'; }
    }
}

async function handleFeedbackSubmit() {
    if (authState.role !== 'employee') return;
    const feedbackInput = document.getElementById('feedbackInput');
    const feedbackStatus = document.getElementById('feedbackStatus');
    const categorySelect = document.getElementById('feedbackCategory');
    const ratingInput = document.querySelector('input[name="rating"]:checked');
    const message = feedbackInput ? feedbackInput.value.trim() : '';

    if (!categorySelect) {
        if (feedbackStatus) { feedbackStatus.textContent = 'Unable to load categories.'; feedbackStatus.className = 'upload-status error'; }
        return;
    }

    if (!ratingInput) {
        if (feedbackStatus) { feedbackStatus.textContent = 'Please select a star rating.'; feedbackStatus.className = 'upload-status error'; }
        return;
    }

    if (!message) {
        if (feedbackStatus) { feedbackStatus.textContent = 'Please enter feedback before submitting.'; feedbackStatus.className = 'upload-status error'; }
        return;
    }
    if (feedbackStatus) { feedbackStatus.innerHTML = '<span class="loading"></span> Sending feedback...'; feedbackStatus.className = 'upload-status'; }

    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ feedback: message, category: categorySelect.value, rating: parseInt(ratingInput.value, 10) })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Unable to submit feedback.');
        if (feedbackInput) feedbackInput.value = '';
        const defaultRating = document.getElementById('star3');
        if (defaultRating) defaultRating.checked = true;
        if (feedbackStatus) { feedbackStatus.textContent = '✅ Feedback shared. HR will review aggregated insights.'; feedbackStatus.className = 'upload-status success'; }
    } catch (error) {
        console.error('Feedback error:', error);
        if (feedbackStatus) { feedbackStatus.textContent = `❌ ${error.message}`; feedbackStatus.className = 'upload-status error'; }
    }
}

async function loadFeedbackSummary() {
    if (authState.role !== 'hr') return;
    const overviewCard = document.getElementById('sentimentOverview');
    const sentimentChart = document.getElementById('sentimentSummaryChart');
    const chartContainer = document.getElementById('departmentChart');
    const categoryChart = document.getElementById('categoryChart');
    const recommendationsList = document.getElementById('sentimentRecommendations');
    const entriesContainer = document.getElementById('recentFeedback');
    if (!overviewCard) return;
    overviewCard.innerHTML = '<span class="loading"></span> Generating insights...';
    if (sentimentChart) sentimentChart.innerHTML = '';
    if (chartContainer) chartContainer.innerHTML = '';
    if (categoryChart) categoryChart.innerHTML = '';
    if (recommendationsList) recommendationsList.innerHTML = '';
    if (entriesContainer) entriesContainer.innerHTML = '';

    try {
        const response = await fetch('/api/hr/feedback_summary', { method: 'GET', headers: buildAuthHeaders() });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Unable to load feedback insights.');
        renderFeedbackDashboard(data);
    } catch (error) {
        console.error('Feedback summary error:', error);
        overviewCard.textContent = `❌ ${error.message}`;
        overviewCard.className = 'feedback-summary-card';
    }
}

function renderFeedbackDashboard(data) {
    const overviewCard = document.getElementById('sentimentOverview');
    const sentimentChart = document.getElementById('sentimentSummaryChart');
    const chartContainer = document.getElementById('departmentChart');
    const categoryChart = document.getElementById('categoryChart');
    const recommendationsList = document.getElementById('sentimentRecommendations');
    const entriesContainer = document.getElementById('recentFeedback');
    if (!overviewCard) return;

    const overview = data.sentiment_overview || {};
    overviewCard.innerHTML = `
        <p><strong>Total Feedback:</strong> ${data.total_feedback || 0}</p>
        <p>${overview.overview || 'No feedback submitted yet.'}</p>
    `;
    if (overview.actions && overview.actions.length) {
        const list = document.createElement('ul');
        list.className = 'helper-text';
        overview.actions.forEach(action => {
            const li = document.createElement('li');
            li.textContent = action.action || action;
            list.appendChild(li);
        });
        overviewCard.appendChild(list);
    }

    if (sentimentChart) {
        sentimentChart.innerHTML = '';
        const totals = data.sentiment_totals || {};
        ['positive', 'neutral', 'negative'].forEach(key => {
            const tile = document.createElement('div');
            tile.className = `sentiment-tile ${key}`;
            const label = document.createElement('h4');
            label.textContent = key.charAt(0).toUpperCase() + key.slice(1);
            const count = document.createElement('div');
            count.className = 'count';
            count.textContent = totals[key] || 0;
            tile.appendChild(label);
            tile.appendChild(count);
            sentimentChart.appendChild(tile);
        });
    }

    if (chartContainer) {
        chartContainer.innerHTML = '';
        const distribution = data.department_distribution || [];
        if (!distribution.length) {
            chartContainer.innerHTML = '<p class="helper-text">Sentiment by department will appear after feedback is received.</p>';
        } else {
            distribution.forEach(row => {
                const wrapper = document.createElement('div');
                wrapper.className = 'dept-row';
                const label = document.createElement('div');
                label.className = 'dept-label';
                label.innerHTML = `<span>${row.department}</span><span>${row.total} responses</span>`;
                const bar = document.createElement('div');
                bar.className = 'dept-bar';
                ['positive', 'neutral', 'negative'].forEach(key => {
                    const span = document.createElement('span');
                    span.className = key;
                    const total = row.total || 1;
                    const pct = Math.round((row[key] / total) * 100);
                    span.style.width = pct + '%';
                    bar.appendChild(span);
                });
                wrapper.appendChild(label);
                wrapper.appendChild(bar);
                chartContainer.appendChild(wrapper);
            });
        }
    }

    if (categoryChart) {
        categoryChart.innerHTML = '';
        const categories = data.category_breakdown || [];
        if (!categories.length) {
            categoryChart.innerHTML = '<p class="helper-text">Category-wise sentiment will appear after feedback is received.</p>';
        } else {
            categories.forEach(cat => {
                const wrapper = document.createElement('div');
                wrapper.className = 'dept-row';
                const label = document.createElement('div');
                label.className = 'dept-label';
                label.innerHTML = `<span>${cat.category}</span><span>${cat.count} entries</span>`;

                const bar = document.createElement('div');
                bar.className = 'dept-bar';
                ['positive', 'neutral', 'negative'].forEach(key => {
                    const span = document.createElement('span');
                    span.className = key;
                    const total = cat.count || 1;
                    span.style.width = Math.round((cat[key] / total) * 100) + '%';
                    bar.appendChild(span);
                });
                wrapper.appendChild(label);
                wrapper.appendChild(bar);
                categoryChart.appendChild(wrapper);
            });
        }
    }

    if (recommendationsList) {
        recommendationsList.innerHTML = '';
        const recs = data.recommendations || [];
        if (recs.length) {
            recs.forEach(rec => {
                const item = document.createElement('div');
                item.className = 'recommendation-item';
                const title = document.createElement('h4');
                title.textContent = rec.title || rec.action || 'Recommendation';
                const detail = document.createElement('p');
                detail.textContent = rec.detail || rec.action || 'Act on this insight.';
                item.appendChild(title);
                item.appendChild(detail);
                recommendationsList.appendChild(item);
            });
        } else {
            recommendationsList.innerHTML = '<p class="helper-text">AI-powered recommendations will appear here.</p>';
        }
    }

    if (entriesContainer) {
        entriesContainer.innerHTML = '';
        const recent = data.recent_feedback || [];
        if (!recent.length) {
            entriesContainer.innerHTML = '<p class="helper-text">Recent feedback will appear here.</p>';
        } else {
            recent.forEach(entry => {
                const item = document.createElement('div');
                item.className = 'feedback-entry';

                const meta = document.createElement('div');
                meta.className = 'meta';
                const date = entry.submitted_at ? new Date(entry.submitted_at).toLocaleString() : 'Recently';
                const rating = entry.rating || 0;
                const stars = '★'.repeat(rating) + '☆'.repeat(5 - rating);
                meta.innerHTML = `
                    <span>${entry.category || 'General'}</span>
                    <span>${entry.department || 'Unknown dept'}</span>
                    <span>${stars}</span>
                    <span>${date}</span>
                `;

                const sentimentChip = document.createElement('span');
                const sentimentLabel = entry.sentiment || 'neutral';
                sentimentChip.className = `sentiment-chip ${sentimentLabel}`;
                sentimentChip.textContent = sentimentLabel;

                const text = document.createElement('div');
                text.textContent = entry.feedback;

                item.appendChild(meta);
                item.appendChild(sentimentChip);
                item.appendChild(text);
                entriesContainer.appendChild(item);
            });
        }
    }
}
