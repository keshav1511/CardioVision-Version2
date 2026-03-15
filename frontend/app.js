// Configuration
const API_URL = 'https://cardiovision-version2.onrender.com';

// Utilities
const showAlert = (alertId, message, type = 'error') => {
    const alertEl = document.getElementById(alertId);
    if (!alertEl) return;
    alertEl.textContent = message;
    alertEl.className = `alert alert-${type}`;
    alertEl.style.display = 'block';
    setTimeout(() => { alertEl.style.display = 'none'; }, 5000);
};

const getToken = () => localStorage.getItem('cardiovision_token');
const setToken = (token) => localStorage.setItem('cardiovision_token', token);
const removeToken = () => localStorage.removeItem('cardiovision_token');
const saveUser = (user) => localStorage.setItem('cardiovision_user', JSON.stringify(user));
const getUser = () => JSON.parse(localStorage.getItem('cardiovision_user') || '{}');

// Routing Logic
const checkAuth = () => {
    const path = window.location.pathname;
    const token = getToken();
    const isAuthPage = path.endsWith('login.html') || path.endsWith('signup.html');

    // Auto-redirect if token is missing
    if (!token && !isAuthPage && !path.endsWith('/') && !path.endsWith('index.html')) {
        window.location.href = 'login.html';
    }
    // Auto-redirect if token exists and on auth page
    else if (token && isAuthPage) {
        window.location.href = 'dashboard.html';
    }
};

// Initialize app based on route
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();

    // Setup specific pages
    const path = window.location.pathname;

    if (path.endsWith('login.html')) setupLogin();
    else if (path.endsWith('signup.html')) setupSignup();
    else if (path.endsWith('dashboard.html')) setupDashboard();

    // Temporary redirect for root to login
    if (path.endsWith('/') || path.endsWith('index.html')) {
        if (getToken()) window.location.href = 'dashboard.html';
        else window.location.href = 'login.html';
    }
});

// ------------- PAGES ------------- //

function setupLogin() {
    const form = document.getElementById('login-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const btn = document.getElementById('login-btn');
        const loader = document.getElementById('login-loader');

        btn.querySelector('span').style.display = 'none';
        loader.style.display = 'inline-block';
        btn.disabled = true;

        try {
            const formData = new URLSearchParams();
            formData.append('username', email); // OAuth2 requires username field
            formData.append('password', password);

            const res = await fetch(`${API_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: formData
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Login failed');

            setToken(data.access_token);
            // In a real app we might fetch user profile here. We'll store dummy for view.
            saveUser({ email, name: email.split('@')[0] });
            window.location.href = 'dashboard.html';
        } catch (err) {
            showAlert('login-alert', err.message);
            btn.querySelector('span').style.display = 'inline-block';
            loader.style.display = 'none';
            btn.disabled = false;
        }
    });
}

function setupSignup() {
    const form = document.getElementById('signup-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('name').value;
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const btn = document.getElementById('signup-btn');
        const loader = document.getElementById('signup-loader');

        btn.querySelector('span').style.display = 'none';
        loader.style.display = 'inline-block';
        btn.disabled = true;

        try {
            const res = await fetch(`${API_URL}/signup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, password })
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Registration failed');

            // Auto-login or redirect
            showAlert('signup-alert', 'Account created successfully! Redirecting...', 'success');
            setTimeout(() => { window.location.href = 'login.html'; }, 1500);
        } catch (err) {
            showAlert('signup-alert', err.message);
            btn.querySelector('span').style.display = 'inline-block';
            loader.style.display = 'none';
            btn.disabled = false;
        }
    });
}

function setupDashboard() {
    // Top nav logic
    const user = getUser();
    const nameEl = document.getElementById('user-display-name');
    if (nameEl) nameEl.textContent = `Welcome, ${user.name || 'Doctor'}`;

    document.getElementById('logout-btn').addEventListener('click', () => {
        removeToken();
        localStorage.removeItem('cardiovision_user');
        window.location.href = 'login.html';
    });

    // File Upload Logic
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input-hidden');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const form = document.getElementById('upload-form');

    // Drag & Drop
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect(fileInput.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFileSelect(fileInput.files[0]);
    });

    function handleFileSelect(file) {
        if (file.size > 10 * 1024 * 1024) {
            showAlert('dashboard-alert', 'File size exceeds 10MB limit.');
            return;
        }
        if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
            showAlert('dashboard-alert', 'Invalid file type. Use JPEG or PNG.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            dropZone.style.display = 'none'; // hide drop zone

            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('btn-disabled');
        };
        reader.readAsDataURL(file);
    }

    // Analyze Form Submission
    let currentPredictionId = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) return;

        const btnSpan = analyzeBtn.querySelector('span');
        const loader = document.getElementById('analyze-loader');

        btnSpan.style.display = 'none';
        loader.style.display = 'inline-block';
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${getToken()}`
                },
                body: formData
            });

            if (res.status === 401) {
                removeToken();
                window.location.href = 'login.html';
                return;
            }

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Prediction failed');

            // Show Results Panel
            document.getElementById('results-panel').classList.remove('hidden');

            // Update UI
            const riskDisplay = document.getElementById('risk-display');
            const confDisplay = document.getElementById('confidence-display');
            const heatmapImg = document.getElementById('heatmap-image');

            riskDisplay.textContent = data.prediction_class;

            // Set styles based on risk
            riskDisplay.className = 'risk-level';
            if (data.prediction_class === 'High Risk') {
                riskDisplay.classList.add('risk-high');
            } else {
                riskDisplay.classList.add('risk-low');
            }

            // Show percentages
            const riskScorePc = (data.risk_score * 100).toFixed(1);
            const confPc = (data.confidence * 100).toFixed(1);

            confDisplay.textContent = `Confidence: ${confPc}% (Score: ${riskScorePc}%)`;

            // Heatmap
            heatmapImg.src = `${API_URL}${data.heatmap_url}`;

            currentPredictionId = data.id;

            // Scroll to results on mobile
            if (window.innerWidth < 992) {
                document.getElementById('results-panel').scrollIntoView({ behavior: 'smooth' });
            }

        } catch (err) {
            showAlert('dashboard-alert', err.message);
        } finally {
            btnSpan.style.display = 'inline-block';
            loader.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    // Download Report
    document.getElementById('download-report-btn').addEventListener('click', async () => {
        if (!currentPredictionId) return;

        try {
            const res = await fetch(`${API_URL}/download-report?prediction_id=${currentPredictionId}`, {
                headers: {
                    'Authorization': `Bearer ${getToken()}`
                }
            });

            if (!res.ok) throw new Error('Failed to generate report');

            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `CardioVision_Report_${currentPredictionId.substring(0, 8)}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

        } catch (err) {
            showAlert('dashboard-alert', err.message);
        }
    });
}
