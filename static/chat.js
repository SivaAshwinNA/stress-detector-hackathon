let socket=null;
let currentRoom = null;
let username = ''; // Will be initialized in DOMContentLoaded
let roomMessages = {};
let firebaseUser = null;
let mediaStream = null;
let frameTimer = null;
let sessionStart = null; // Track session start time

function initSocket(){
    console.log('initSocket called.'); // Debugging line
    if (socket) return;
    socket = io();
    socket.on('connect', () => {
        console.log('Socket connected!'); // Debugging line
        const defaultRoom = getFirstRoomName() || '🧠 StressScope';
        joinRoom(defaultRoom);
        highlightActiveRoom(defaultRoom);
        // Set session start time when first connecting
        if (!sessionStart) {
            sessionStart = new Date().toISOString();
            console.log('Session started at:', sessionStart);
        }
    });
    socket.on('message', (data) => {
        addMessage(
            data.username,
            data.msg,
            data.username === username ? 'own' : 'other'
        );
    });
    socket.on('private_message', (data) => {
        addMessage(data.from, `[Private] ${data.msg}`, 'private');
    });
    socket.on('status', (data) => {
        addMessage('System', data.msg, 'system');
    });
    socket.on('active_users', (data) => {
        const userList = document.getElementById('active-users');
        userList.innerHTML = data.users
            .map(
                (user) => `
            <div class="user-item" onclick="insertPrivateMessage('${user}')">
                ${user} ${user === username ? '(you)' : ''}
            </div>
        `
            )
            .join('');
    });
}

// Message Handling
function addMessage(sender, message, type){
    if(!roomMessages[currentRoom]){
        roomMessages[currentRoom] = [];
    }
    roomMessages[currentRoom].push({ sender, message, type });

    const chat = document.getElementById('chat');
    const messageDiv = document.createElement('div');

    messageDiv.className = `message ${type}`;
    messageDiv.textContent = `${sender}: ${message}`;

    chat.appendChild(messageDiv);
    chat.scrollTop = chat.scrollHeight;
}

function sendMessage(){
    const input = document.getElementById('message');
    const message = input.value.trim();

    if(!message) return;

    // Require login before sending
    if (!username || username.startsWith('Guest')){
        alert('Please log in to start chatting.');
        openLoginOverlay();
        return;
    }

    if(message.startsWith('@')) {
        const [target, ...msgParts]=message.substring(1).split(' ');
        const privateMsg = msgParts.join(' ');

        if(privateMsg){
            socket.emit('message',{
                msg:privateMsg,
                type:'private',
                target:target,
            });
        }
    }else{
        socket.emit('message',{
            msg: message,
            room: currentRoom,
        });
    }

    input.value='';
    input.focus();
}

// Join the room
function joinRoom(room){
    console.log(`Attempting to join room: ${room}`); // Debugging line
    if (currentRoom && currentRoom !== room) {
        socket.emit('leave', { room: currentRoom });
    }
    currentRoom = room;
    socket.emit('join', { room });
    // update active room highlight whenever we switch rooms
    highlightActiveRoom(currentRoom);

    const chat = document.getElementById('chat');
    chat.innerHTML = '';

    if (roomMessages[room]){
        roomMessages[room].forEach((msg) => {
            addMessage(msg.sender, msg.message, msg.type);
        });
    }
}

// Insert Private Message
function insertPrivateMessage(user) {
	document.getElementById('message').value = `@${user} `;
	document.getElementById('message').focus();
}

function handleKeyPress(event) {
	if (event.key === 'Enter' && !event.shiftKey) {
		event.preventDefault();
		sendMessage();
	}
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize username from DOM element
    const usernameElement = document.getElementById('username');
    if (usernameElement) {
        username = usernameElement.textContent.trim();
    }
    console.log('Initial username:', username); // Debugging line
    
    if ('Notification' in window) {
        Notification.requestPermission();
    }
    
    // Add click event listener to the join chat button
    const joinBtn = document.getElementById('join-chat-btn');
    if (joinBtn) {
        joinBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Join Chat button clicked'); // Debugging line
            submitLogin();
        });
    }
    
    // Force login overlay if username is not valid (empty or still Jinja placeholder)
    if (!username || username === '' || username === '{{ username }}'){
        console.log('Username is empty or invalid, showing login overlay'); // Debugging line
        openLoginOverlay();
        setTimeout(()=>{
            const inp = document.getElementById('login-username');
            if (inp) inp.focus();
        }, 50);
    } else {
        console.log('Username is valid, initializing socket'); // Debugging line
        initSocket();
    }
});

// Add this new function to handle room highlighting
function highlightActiveRoom(room) {
	document.querySelectorAll('.room-item').forEach((item) => {
		item.classList.remove('active-room');
		if (item.textContent.trim() === room) {
			item.classList.add('active-room');
		}
	});
}

function getFirstRoomName() {
	const first = document.querySelector('.room-item');
	return first ? first.textContent.trim() : null;
}

// --- Simple name overlay functions ---
function openLoginOverlay(){
    console.log('openLoginOverlay called'); // Debugging line
    const o = document.getElementById('login-overlay');
    if (o) {
        o.style.display = 'block';
        console.log('Login overlay should now be visible'); // Debugging line
    } else {
        console.error('Login overlay element not found!'); // Debugging line
    }
}

function closeLoginOverlay(){
    const o = document.getElementById('login-overlay');
    if (o) o.style.display = 'none';
}

async function submitLogin(){
    console.log('submitLogin called'); // Debugging line
    const input = document.getElementById('login-username');
    const val = (input && input.value ? input.value : '').trim();
    console.log('Username input value:', val); // Debugging line
    if (!val){ alert('Please enter a username'); return; }
    try{
        console.log('Sending login request...'); // Debugging line
        const resp = await fetch('/login',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ username: val })
        });
        const data = await resp.json();
        console.log('Login response:', data); // Debugging line
        if (data && data.ok){
            username = data.username;
            const u = document.getElementById('username');
            if (u) u.textContent = username;
            closeLoginOverlay();
            console.log('Login successful, initializing socket...'); // Debugging line
            initSocket();
        } else {
            alert(data.error || 'Login failed');
        }
    }catch(e){
        console.error('Login error:', e); // Debugging line
        alert('Login error');
    }
}

document.addEventListener('keydown', (e)=>{
    const overlay = document.getElementById('login-overlay');
    if (overlay && overlay.style.display !== 'none'){
        if (e.key === 'Enter'){
            e.preventDefault();
            submitLogin();
        }
    }
});

// Removed Google auth functions

// --- Camera capture at ~1 FPS ---
async function toggleCamera(){
    const btn = document.getElementById('camera-toggle');
    const status = document.getElementById('camera-status');
    const preview = document.getElementById('preview');
    if (!mediaStream){
        try{
            mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            preview.srcObject = mediaStream;
            preview.style.display = 'block';
            btn.textContent = 'Stop Camera';
            status.textContent = 'Camera on (1 FPS)';
            startFrameLoop();
        }catch(err){
            console.error('Camera permission denied or error', err);
            status.textContent = 'Camera error';
        }
    }else{
        stopFrameLoop();
        mediaStream.getTracks().forEach(t=>t.stop());
        mediaStream = null;
        preview.srcObject = null;
        preview.style.display = 'none';
        btn.textContent = 'Start Camera (consent)';
        status.textContent = 'Camera off';
    }
}

function startFrameLoop(){
    if (frameTimer) return;
    const preview = document.getElementById('preview');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    frameTimer = setInterval(async ()=>{
        if (!preview.videoWidth || !preview.videoHeight) return;
        canvas.width = preview.videoWidth;
        canvas.height = preview.videoHeight;
        ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(res=>canvas.toBlob(res, 'image/jpeg', 0.7));
        if (!blob) return;
        const form = new FormData();
        form.append('frame', blob, `frame_${Date.now()}.jpg`);
        try{
            await fetch('/upload_frame', { method:'POST', body: form });
        }catch(e){
            console.error('Failed to upload frame', e);
        }
    }, 1000); // 1 FPS
}

function stopFrameLoop(){
    if (frameTimer){
        clearInterval(frameTimer);
        frameTimer = null;
    }
}

// --- Analysis fetch and Chart.js rendering ---
let chatChart=null;
let lastAnalysisPayload=null;
let analysisAbortController=null;
let analysisProgressTimer=null;
async function fetchAnalysis(){
    const btn = document.getElementById('run-analysis-btn');
    const stat = document.getElementById('analysis-status');
    if (btn) btn.disabled = true; 
    if (stat) stat.textContent = 'Running analysis...';
    startProgressPolling();
    try{
        analysisAbortController = new AbortController();
        // Send current room and session start time
        const since = sessionStart || new Date(Date.now() - 60*60*1000).toISOString();
        console.log('Sending analysis request with room:', currentRoom, 'since:', since);
        const resp = await fetch('/analysis',{
            method:'POST',
            headers:{ 'Content-Type':'application/json' },
            body: JSON.stringify({ room: currentRoom, since }),
            signal: analysisAbortController.signal
        });
        const data = await resp.json();
        console.log('Analysis response:', data); // Debug logging
        lastAnalysisPayload = data;
        renderCharts(data);
        renderRecommendations(data.recommendations || []);
        if (stat) stat.textContent = 'Done';
        const dl = document.getElementById('download-analysis');
        if (dl) dl.style.display='inline-block';
    }catch(e){
        console.error('Analysis error:', e); // Debug logging
        if (e.name === 'AbortError'){
            if (stat) stat.textContent = 'Analysis cancelled';
        } else {
            if (stat) stat.textContent = 'Error running analysis';
        }
    }finally{
        if (btn) btn.disabled = false;
        analysisAbortController = null;
        stopProgressPolling(true);
    }
}

function lineCfg(label, labels, values, color){
    return {
        type: 'line',
        data: {
            labels,
            datasets: [{ label, data: values, borderColor: color, tension: 0.2, pointRadius: 1 }]
        },
        options: {
            animation: false,
            scales: { 
                x: {
                    type: 'time',
                    time: { unit: 'minute' },
                    adapters: { date: Chart.adapters.dateFns }
                },
                y: { min:0, max:100 } 
            },
            plugins: { legend: { display: true } }
        }
    };
}

function renderCharts(payload){
    console.log('renderCharts called with payload:', payload); // Debug logging
    const chatSeries = (payload.chat_series||[]).sort((a,b)=>a.t.localeCompare(b.t));
    const videoSeries = (payload.video_series||[]).sort((a,b)=>a.t.localeCompare(b.t));
    const combinedSeries = combineSeries(chatSeries, videoSeries);
    console.log('Chat series length:', chatSeries.length); // Debug logging
    console.log('Video series length:', videoSeries.length); // Debug logging
    
    // Handle case when there's no data
    if (chatSeries.length === 0 && videoSeries.length === 0) {
        console.log('No data available for charts');
        renderCards(payload);
        return;
    }

    // Destroy existing charts
    if (chatChart) chatChart.destroy();
    // Only chat chart is used now

    // Check if canvas elements exist
    const chatCanvas = document.getElementById('chatChart');
    const videoCanvas = null;
    const combinedCanvas = null;
    
    if (!chatCanvas) {
        console.error('Chart canvas elements not found!');
        return;
    }

    // Prepare data for charts - use proper format for Chart.js
    // For chat chart, use labels + numeric data for maximum compatibility
    const chatLabels = chatSeries.map(p => new Date(p.t));
    const chatValues = chatSeries.map(p => p.score);
    // For other charts, keep x/y pairs
    const chatData = chatSeries.map(p => ({ x: new Date(p.t), y: p.score }));
    // Pruned
    
    console.log('Chat data sample:', chatData.slice(0, 3)); // Debug logging
    // Pruned
    
    // Validate data before creating charts
    if (chatData.some(d => isNaN(d.x) || isNaN(d.y))) {
        console.error('Invalid chat data detected');
    }
    // Pruned

    // Chat Chart - Line graph
    try {
        const chatCtx = chatCanvas.getContext('2d');
        chatChart = new Chart(chatCtx, {
            type: 'line',
            data: { 
                labels: chatLabels,
                datasets: [{ 
                    label: 'Chat Stress Level', 
                    data: chatValues, 
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4, 
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    fill: true
                }]
            },
            options: { 
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { 
                        type: 'time',
                        min: payload.chat_time?.start ? new Date(payload.chat_time.start) : undefined,
                        max: payload.chat_time?.end ? new Date(payload.chat_time.end) : undefined,
                        time: {
                            displayFormats: { minute: 'HH:mm', hour: 'HH:mm' }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: { 
                        min: 0, 
                        max: 100,
                        title: {
                            display: true,
                            text: 'Stress Level (%)'
                        }
                    }
                },
                plugins: { 
                    legend: { display: true },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Stress: ${context.parsed.y}%`;
                            }
                        }
                    }
                }
            }
        });
        console.log('Chat chart created successfully');
    } catch (error) {
        console.error('Error creating chat chart:', error);
    }

    // Pruned video and combined charts

    renderCards(payload);
}

function combineSeries(a, b){
    // naive time-merge: for each unique timestamp, average available series
    const map = new Map();
    a.forEach(p=>{ map.set(p.t, {sum:p.score, n:1}); });
    b.forEach(p=>{
        const v = map.get(p.t);
        if (v) { v.sum += p.score; v.n += 1; }
        else { map.set(p.t, {sum:p.score, n:1}); }
    });
    return Array.from(map.entries()).sort((x,y)=>x[0].localeCompare(y[0])).map(([t,v])=>({ t, score: Math.round(v.sum/v.n) }));
}

function renderRecommendations(recs){
    const container = document.getElementById('recommendations');
    if (!recs || !recs.length){ container.innerHTML = ''; return; }
    container.innerHTML = '<h3>Recommendations</h3>' + recs.map(r=>`<div><a href="${r.link}" target="_blank">${r.title}</a></div>`).join('');
}

function renderCards(payload){
    console.log('renderCards called with payload:', payload); // Debug logging
    // Chat cards
    const cwrap = document.getElementById('chat-cards');
    if (cwrap){
        const avgV = (payload.chat_metrics?.avg ?? 0).toFixed(0);
        const cnt = payload.chat_metrics?.count_messages ?? 0;
        const peak = payload.chat_metrics?.peak?.score ?? '-';
        console.log('Chat metrics:', { avgV, cnt, peak }); // Debug logging
        cwrap.innerHTML = card('Avg Stress', avgV+'%') + card('Messages', cnt) + card('Peak Stress', peak);
    } else {
        console.error('chat-cards element not found!');
    }
    // Pruned video cards
    // Combined card to include categorical stress level
    const combinedWrap = document.getElementById('combined-card'); // Assuming an element with this ID exists or will be created in chat.html
    if (combinedWrap && payload.combined_level_text){
        combinedWrap.innerHTML = card('Overall Stress Level', payload.combined_level_text);
    }
}

function card(title, value){
    return `<div style="flex:1;min-width:140px;background:white;border:1px solid #eef2f7;border-radius:8px;padding:8px 10px;">
      <div style="font-size:12px;color:#6b7280">${title}</div>
      <div style="font-weight:600;color:#111827;font-size:16px;">${value}</div>
    </div>`;
}

function avg(arr){ if (!arr || !arr.length) return 0; return Math.round(arr.reduce((a,b)=>a+b,0)/arr.length); }

function downloadAnalysis(){
    if (!lastAnalysisPayload) return;
    const { jsPDF } = window.jspdf || {};
    if (!jsPDF){
        alert('PDF generator not available');
        return;
    }
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });
    const margin = 40;
    let y = margin;
    const line = (text, size=12, bold=false)=>{
        doc.setFont('helvetica', bold ? 'bold' : 'normal');
        doc.setFontSize(size);
        doc.text(String(text), margin, y);
        y += size + 8;
    };
    // Title
    line('Chat Stress Analysis', 18, true);
    // Time range
    if (lastAnalysisPayload.chat_time){
        line(`Period: ${lastAnalysisPayload.chat_time.start || '-'} to ${lastAnalysisPayload.chat_time.end || '-'}`, 10);
    }
    // Cards
    const cm = lastAnalysisPayload.chat_metrics || {};
    line(`Avg Stress: ${cm.avg != null ? Math.round(cm.avg) + '%' : '-'}`, 12);
    line(`Messages: ${cm.count_messages ?? 0}`, 12);
    line(`Peak Stress: ${cm.peak?.score ?? '-'}`, 12);
    // Overall level
    if (lastAnalysisPayload.combined_level_text){
        line(`Overall Stress Level: ${lastAnalysisPayload.combined_level_text}`, 12, true);
    }
    // Recommendations
    const recs = lastAnalysisPayload.recommendations || [];
    if (recs.length){
        line('Recommendations:', 14, true);
        recs.forEach((r, idx)=>{
            const title = r.title || 'Recommendation';
            const link = r.link || '';
            line(`${idx+1}. ${title}`, 11);
            if (link){ line(link, 10); }
        });
    }
    doc.save(`analysis_${Date.now()}.pdf`);
}

async function endSession(){
    if (mediaStream){ toggleCamera(); }
    if (currentRoom){ socket.emit('leave', { room: currentRoom }); }
    try { socket.disconnect(); } catch {}
    openAnalysisOverlay();
}

function openAnalysisOverlay(){
    const overlay = document.getElementById('analysis-overlay');
    if (overlay) {
        overlay.style.display = 'block';
        // Give the overlay a tick to layout, then resize charts
        setTimeout(()=>{
            try {
                if (chatChart) { chatChart.resize(); chatChart.update('none'); }
                if (videoChart) { videoChart.resize(); videoChart.update('none'); }
                if (combinedChart) { combinedChart.resize(); combinedChart.update('none'); }
            } catch(e) { console.warn('Chart resize error', e); }
        }, 50);
    }
}

function closeAnalysisOverlay(){
    const overlay = document.getElementById('analysis-overlay');
    if (overlay) overlay.style.display = 'none';
}

async function runAndShowAnalysis(){
    const stat = document.getElementById('analysis-status');
    const runBtn = document.getElementById('run-analysis-btn');
    const cancelBtn = document.getElementById('cancel-analysis-btn');
    if (runBtn) runBtn.style.display='none';
    if (cancelBtn) cancelBtn.style.display='inline-block';
    if (stat) stat.textContent = 'Running analysis...';
    await fetchAnalysis();
    if (runBtn) runBtn.style.display='inline-block';
    if (cancelBtn) cancelBtn.style.display='none';
}

function cancelAnalysis(){
    const stat = document.getElementById('analysis-status');
    if (analysisAbortController){
        analysisAbortController.abort();
        analysisAbortController = null;
    }
    if (stat) stat.textContent = 'Cancelling analysis...';
    stopProgressPolling(false);
}

function startProgressPolling(){
    stopProgressPolling(false);
    analysisProgressTimer = setInterval(async ()=>{
        try{
            const r = await fetch('/analysis/progress');
            const j = await r.json();
            const pct = Math.max(0, Math.min(100, parseInt(j.percent||0)));
            const bar = document.getElementById('analysis-progress-bar');
            const num = document.getElementById('analysis-percent');
            if (bar) bar.style.width = pct + '%';
            if (num) num.textContent = pct + '%';
        }catch{}
    }, 300);
}

function stopProgressPolling(forceDone){
    if (analysisProgressTimer){
        clearInterval(analysisProgressTimer);
        analysisProgressTimer = null;
    }
    if (forceDone){
        const bar = document.getElementById('analysis-progress-bar');
        const num = document.getElementById('analysis-percent');
        if (bar) bar.style.width = '100%';
        if (num) num.textContent = '100%';
    }
}