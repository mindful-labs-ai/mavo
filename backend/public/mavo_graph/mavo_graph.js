// Chart and graph rendering functions for Mavo
// This file is a module and does not use or set any global variables from mavo.js
// All functions take data as arguments and render into the provided DOM elements

// let is_verbose_graph = false;
window.is_verbose_graph = window.is_verbose_graph || false;

// Function to run sentiment analysis
async function runSentimentAnalysis(resultContainerId = 'analysis-results', baseUrl = '') {
    if (!currentAudioUuid) {
        if (typeof showError === 'function') {
            showError("No audio file loaded. Please upload an audio file first.");
        }
        return;
    }
    let sentimentBtn = null;
    try {
        // Disable the button during processing
        if (sentimentBtn) {
            sentimentBtn.disabled = true;
            sentimentBtn.textContent = "분석중...";
        }
        
        // Show loading spinner
        if (typeof showSpinner === 'function') {
            showSpinner("분석중...");
        }
        
        // Clear any existing analysis results
        const analysisResults = document.getElementById(resultContainerId);
        analysisResults.innerHTML = '';
        
        // Fixed API path - use absolute path instead of appending to API_BASE_URL
        const maumApiUrl = `${baseUrl}/api/v1/maum/${currentAudioUuid}`;
        
        // Make API call to run sentiment analysis
        const response = await fetch(maumApiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to run sentiment analysis');
        }
        
        // Process the result
        const result = await response.json();
        
        // Hide spinner
        if (typeof hideSpinner === 'function') {
            hideSpinner();
        }
        
        // Show success message
        if (typeof showSuccess === 'function') {
            showSuccess("Analysis completed successfully");
        }
        
        // Automatically show interactive analysis in the analysis-results area
        await showInteractiveAnalysis(resultContainerId, baseUrl);
        
    } catch (error) {
        if (typeof hideSpinner === 'function') {
            hideSpinner();
        }
        if (typeof showError === 'function') {
            showError(`Error running sentiment analysis: ${error.message}`);
        }
        if (is_verbose_graph) {
        console.error('Error running sentiment analysis:', error);
        }
        throw error;
    } finally {
        // Re-enable the button
        if (sentimentBtn) {
            sentimentBtn.disabled = false;
            sentimentBtn.textContent = "감정분석1";
        }
    }
}


// Interactive Analysis Functions
async function showInteractiveAnalysis(resultContainerId = 'analysis-results', baseUrl = '') {
    if (!currentAudioUuid) {
        if (typeof showError === 'function') {
            showError("No audio file loaded. Please upload an audio file first.");
        }
        return;
    }
        
    try {
        // Show loading spinner
        if (typeof showSpinner === 'function') {
            showSpinner("분석중...");
        }
        
        // Get the analysis data from the server
        const response = await fetch(`${baseUrl}/api/v1/maum/data/${currentAudioUuid}`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to get analysis data');
        }
        
        const result = await response.json();
        
        // Hide spinner
        if (typeof hideSpinner === 'function') {
            hideSpinner();
        }
        
        // Clear any existing content in the analysis results area
        const analysisResults = document.getElementById('analysis-results');
        analysisResults.innerHTML = '';
        
        // Add statistics section
        const data = result.data;
        const wordSegments = data.word_segments || [];
        const segments = data.consecutive_segments || [];
        const summary = data.summary || {};
        
        // Calculate total time
        let totalTime = 0;
        let counselorTime = 0;
        let clientTimes = {1: 0, 2: 0, 3: 0, 4: 0};
        let silenceTime = 0;
        
        if (wordSegments.length > 0) {
            totalTime = wordSegments[wordSegments.length - 1].end;
            
            // Calculate talking times
            if (summary.talking_times) {
                counselorTime = summary.talking_times["0"] || 0;
                for (let role = 1; role <= 4; role++) {
                    clientTimes[role] = summary.talking_times[role.toString()] || 0;
                }
            }
            
            // Calculate silence time
            const totalClientTime = Object.values(clientTimes).reduce((sum, time) => sum + time, 0);
            silenceTime = totalTime - counselorTime - totalClientTime;
            if (silenceTime < 0) silenceTime = 0;
        }
        
        // Format keywords as combined nouns (regular + proper), filtering out short words, adding frequencies, and sorting by frequency
        let mainNouns = '';
        
        if (summary.frequencies) {
            let combinedNouns = [];
            
            // Add regular nouns (NNG)
            if (summary.frequencies.NNG && summary.frequencies.NNG.length > 0) {
                combinedNouns = combinedNouns.concat(
                    summary.frequencies.NNG
                        .filter(item => item[0].length >= 2 && item[1] >= 3)  // Filter out words shorter than 2 chars or fewer than 3 occurrences
                        .map(item => ({
                            word: item[0],
                            frequency: item[1],
                            isProper: false
                        }))
                );
            }
            
            // Add proper nouns (NNP)
            if (summary.frequencies.NNP && summary.frequencies.NNP.length > 0) {
                combinedNouns = combinedNouns.concat(
                    summary.frequencies.NNP
                        .filter(item => item[0].length >= 2 && item[1] >= 3)  // Filter out words shorter than 2 chars or fewer than 3 occurrences
                        .map(item => ({
                            word: item[0],
                            frequency: item[1],
                            isProper: true
                        }))
                );
            }
            
            // Sort by frequency (descending)
            combinedNouns.sort((a, b) => b.frequency - a.frequency);
            
            // Convert to HTML elements with appropriate styling - use grey background for keywords
            mainNouns = combinedNouns
                .map(item => {
                    // Use grey background for all keywords
                    const className = item.isProper ? 'proper-noun-value' : 'noun-value';
                    return `<span class="${className}" style="background-color: #f0f0f0;">${item.word}<span class="noun-freq">(${item.frequency})</span></span>`;
                })
                .join(' ');
        }
        
        // Calculate percentages for talking time
        const counselorPercent = totalTime > 0 ? Math.round((counselorTime / totalTime) * 100) : 0;
        const clientPercents = {};
        for (let role = 1; role <= 4; role++) {
            clientPercents[role] = totalTime > 0 ? Math.round((clientTimes[role] / totalTime) * 100) : 0;
        }
        const silencePercent = totalTime > 0 ? Math.round((silenceTime / totalTime) * 100) : 0;
        
        // Get cadence values
        const counselorCadence = summary.avg_cadence ? Math.round(summary.avg_cadence["0"] || 0) : 0;
        const clientCadences = {};
        for (let role = 1; role <= 4; role++) {
            clientCadences[role] = summary.avg_cadence ? Math.round(summary.avg_cadence[role.toString()] || 0) : 0;
        }
        
        // Get sentiment values and calculate percentages
        let counselorPosCount = 0;
        let counselorNegCount = 0;
        let counselorNeutralCount = 0;
        
        // Initialize sentiment counts for each client role
        const clientPosCounts = {1: 0, 2: 0, 3: 0, 4: 0};
        const clientNegCounts = {1: 0, 2: 0, 3: 0, 4: 0};
        const clientNeutralCounts = {1: 0, 2: 0, 3: 0, 4: 0};
        
        // Analyze sentiment for counselor
        const counselorSegments = segments.filter(s => s.speaker_role === 0);
        
        counselorSegments.forEach(segment => {
            if (segment.sentiment) {
                const pos = segment.sentiment.pos_senti || 0;
                const neg = segment.sentiment.neg_senti || 0;
                
                if (pos > 0.2 && pos > neg) {  // Threshold for positive
                    counselorPosCount++;
                } else if (neg > 0.2 && neg > pos) {  // Threshold for negative
                    counselorNegCount++;
                } else {
                    counselorNeutralCount++;
                }
            } else {
                counselorNeutralCount++;
            }
        });
        
        // Analyze sentiment for each client role
        const clientSegmentsByRole = {};
        for (let role = 1; role <= 4; role++) {
            clientSegmentsByRole[role] = segments.filter(s => s.speaker_role === role);
            
            clientSegmentsByRole[role].forEach(segment => {
                if (segment.sentiment) {
                    const pos = segment.sentiment.pos_senti || 0;
                    const neg = segment.sentiment.neg_senti || 0;
                    
                    if (pos > 0.2 && pos > neg) {  // Threshold for positive
                        clientPosCounts[role]++;
                    } else if (neg > 0.2 && neg > pos) {  // Threshold for negative
                        clientNegCounts[role]++;
                    } else {
                        clientNeutralCounts[role]++;
                    }
                } else {
                    clientNeutralCounts[role]++;
                }
            });
        }
        
        // Calculate percentages for sentiment
        const counselorTotal = counselorSegments.length || 1;  // Use total segments count
        const counselorPosPercent = Math.round((counselorPosCount / counselorTotal) * 100);
        const counselorNegPercent = Math.round((counselorNegCount / counselorTotal) * 100);
        const counselorNeutralPercent = Math.round((counselorNeutralCount / counselorTotal) * 100);
        
        // Calculate sentiment percentages for each client role
        const clientPosPercents = {}, clientNegPercents = {}, clientNeutralPercents = {};
        for (let role = 1; role <= 4; role++) {
            const roleTotal = clientSegmentsByRole[role]?.length || 1;
            clientPosPercents[role] = Math.round((clientPosCounts[role] / roleTotal) * 100);
            clientNegPercents[role] = Math.round((clientNegCounts[role] / roleTotal) * 100);
            clientNeutralPercents[role] = Math.round((clientNeutralCounts[role] / roleTotal) * 100);
        }
        
        // Get tense values and calculate percentages
        let counselorFutCount = 0;
        let counselorPastCount = 0;
        let counselorPresentCount = 0;
        
        // Initialize tense counts for each client role
        const clientFutCounts = {1: 0, 2: 0, 3: 0, 4: 0};
        const clientPastCounts = {1: 0, 2: 0, 3: 0, 4: 0};
        const clientPresentCounts = {1: 0, 2: 0, 3: 0, 4: 0};
        
        // Analyze tense for counselor
        counselorSegments.forEach(segment => {
            if (segment.tense) {
                const fut = segment.tense.fut_tense || 0;
                const past = segment.tense.pas_tense || 0;
                
                if (fut > 0.2 && fut > past) {  // Threshold for future
                    counselorFutCount++;
                } else if (past > 0.2 && past > fut) {  // Threshold for past
                    counselorPastCount++;
                } else {
                    counselorPresentCount++;
                }
            } else {
                counselorPresentCount++;
            }
        });
        
        // Analyze tense for each client role
        for (let role = 1; role <= 4; role++) {
            const roleSegments = clientSegmentsByRole[role] || [];
            
            roleSegments.forEach(segment => {
                if (segment.tense) {
                    const fut = segment.tense.fut_tense || 0;
                    const past = segment.tense.pas_tense || 0;
                    
                    if (fut > 0.2 && fut > past) {  // Threshold for future
                        clientFutCounts[role]++;
                    } else if (past > 0.2 && past > fut) {  // Threshold for past
                        clientPastCounts[role]++;
                    } else {
                        clientPresentCounts[role]++;
                    }
                } else {
                    clientPresentCounts[role]++;
                }
            });
        }
        
        // Calculate percentages for tense
        const counselorTenseTotal = counselorSegments.length || 1;
        const counselorFutPercent = Math.round((counselorFutCount / counselorTenseTotal) * 100);
        const counselorPastPercent = Math.round((counselorPastCount / counselorTenseTotal) * 100);
        const counselorPresentPercent = Math.round((counselorPresentCount / counselorTenseTotal) * 100);
        
        // Calculate tense percentages for each client role
        const clientFutPercents = {}, clientPastPercents = {}, clientPresentPercents = {};
        for (let role = 1; role <= 4; role++) {
            const roleTotal = clientSegmentsByRole[role]?.length || 1;
            clientFutPercents[role] = Math.round((clientFutCounts[role] / roleTotal) * 100);
            clientPastPercents[role] = Math.round((clientPastCounts[role] / roleTotal) * 100);
            clientPresentPercents[role] = Math.round((clientPresentCounts[role] / roleTotal) * 100);
        }
        
        // First define the speaker colors consistently
        // Define the same speaker role colors as used in the charts
        const speakerColors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'];

        // Create HTML for the statistics table
        let statsHTML = `
            <h3 style="font-size: 1.3em;">대화 통계</h3>
            <div class="statistics-container">
                <table class="statistics-table">
                    <tr>
                        <td class="stat-label">대화 시간 비율</td>
                        <td class="stat-value">
                            <div class="inline-stats">
                                <div class="speaker-group">
                                    <span class="speaker-label">상담사:</span>
                                    <span class="time-value" style="background-color: ${speakerColors[0]}; color: white;">${counselorPercent}%</span>
                                </div>
        `;
        
        // Count active speaker roles
        const activeSpeakerRoles = Object.keys(clientSegmentsByRole)
            .filter(role => clientSegmentsByRole[role].length > 0)
            .map(role => parseInt(role));
        
        // Add client percentages for each active role
        for (let role = 1; role <= 4; role++) {
            if (clientSegmentsByRole[role]?.length > 0) {
                const clientLabel = getClientLabel(role, activeSpeakerRoles);
                // Use role for color index (roles 1-4 map to array indices 1-4)
                const colorIndex = Math.min(role, speakerColors.length - 1);
                statsHTML += `
                                <div class="speaker-group">
                                    <span class="speaker-label">${clientLabel}:</span>
                                    <span class="time-value" style="background-color: ${speakerColors[colorIndex]}; color: white;">${clientPercents[role]}%</span>
                                </div>`;
            }
        }
        
        // Add silence percentage
        statsHTML += `
                                <div class="speaker-group">
                                    <span class="speaker-label">공백:</span>
                                    <span class="silence-value">${silencePercent}%</span>
                                </div>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td class="stat-label">평균 발화 속도</td>
                        <td class="stat-value">
                            <div class="inline-stats">
                                <div class="speaker-group">
                                    <span class="speaker-label">상담사:</span>
                                    <span class="cadence-value" style="background-color: ${speakerColors[0]}; color: white;">${counselorCadence} <span title="WPM(Word per Minute) - 분당 발화 단어 수">WPM</span></span>
                                </div>
        `;
        
        // Add client cadences for each active role
        for (let role = 1; role <= 4; role++) {
            if (clientSegmentsByRole[role]?.length > 0) {
                const clientLabel = getClientLabel(role, activeSpeakerRoles);
                // Use role for color index (roles 1-4 map to array indices 1-4)
                const colorIndex = Math.min(role, speakerColors.length - 1);
                statsHTML += `
                                <div class="speaker-group">
                                    <span class="speaker-label">${clientLabel}:</span>
                                    <span class="cadence-value" style="background-color: ${speakerColors[colorIndex]}; color: white;">${clientCadences[role]} <span title="WPM(Word per Minute) - 분당 발화 단어 수">WPM</span></span>
                                </div>`;
            }
        }
        
        statsHTML += `
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td class="stat-label">감정 분석</td>
                        <td class="stat-value">
                            <div class="speaker-stats">
                                <span class="speaker-label">상담사:</span>
                                <span class="sentiment-value sentiment-positive">긍정 ${counselorPosPercent}%</span>
                                <span class="sentiment-value sentiment-negative">부정 ${counselorNegPercent}%</span>
                                <span class="sentiment-value sentiment-neutral">중립 ${counselorNeutralPercent}%</span>
                            </div>
        `;
        
        // Add client sentiment for each active role
        for (let role = 1; role <= 4; role++) {
            if (clientSegmentsByRole[role]?.length > 0) {
                const clientLabel = getClientLabel(role, activeSpeakerRoles);
                statsHTML += `
                            <div class="speaker-stats">
                                <span class="speaker-label">${clientLabel}:</span>
                                <span class="sentiment-value sentiment-positive">긍정 ${clientPosPercents[role]}%</span>
                                <span class="sentiment-value sentiment-negative">부정 ${clientNegPercents[role]}%</span>
                                <span class="sentiment-value sentiment-neutral">중립 ${clientNeutralPercents[role]}%</span>
                            </div>
                `;
            }
        }
        
        statsHTML += `
                        </td>
                    </tr>
                    <tr>
                        <td class="stat-label">시제 분석</td>
                        <td class="stat-value">
                            <div class="speaker-stats">
                                <span class="speaker-label">상담사:</span>
                                <span class="tense-value tense-future">미래 ${counselorFutPercent}%</span>
                                <span class="tense-value tense-past">과거 ${counselorPastPercent}%</span>
                                <span class="tense-value tense-present">중립 ${counselorPresentPercent}%</span>
                            </div>
        `;
        
        // Add client tense for each active role
        for (let role = 1; role <= 4; role++) {
            if (clientSegmentsByRole[role]?.length > 0) {
                const clientLabel = getClientLabel(role, activeSpeakerRoles);
                statsHTML += `
                            <div class="speaker-stats">
                                <span class="speaker-label">${clientLabel}:</span>
                                <span class="tense-value tense-future">미래 ${clientFutPercents[role]}%</span>
                                <span class="tense-value tense-past">과거 ${clientPastPercents[role]}%</span>
                                <span class="tense-value tense-present">중립 ${clientPresentPercents[role]}%</span>
                            </div>
                `;
            }
        }
        
        statsHTML += `
                        </td>
                    </tr>
                    <tr>
                        <td class="stat-label">주요 키워드</td>
                        <td class="stat-value">${mainNouns || '<span class="noun-value">없음</span>'}</td>
                    </tr>
                </table>
            </div>
            
            <h3 style="font-size: 1.3em;">대화 차트</h3>
            <div class="charts-grid">
                <div class="chart-card">
                    <h4>발화 속도 (분당 단어 수)</h4>
                    <div id="speechCadenceChart" class="chart-container"></div>
                </div>
                <div class="chart-card">
                    <h4>발화 타임라인</h4>
                    <div id="talkingRatioChart" class="chart-container"></div>
                </div>
                <div class="chart-card">
                    <h4>침묵 구간 (초)</h4>
                    <div id="silenceMomentsChart" class="chart-container"></div>
                </div>
                <div class="chart-card">
                    <h4>감정 분석 (긍정/부정)</h4>
                    <div id="sentimentChart" class="chart-container"></div>
                </div>
                <div class="chart-card">
                    <h4>시간 지향성 (미래/과거)</h4>
                    <div id="tenseChart" class="chart-container"></div>
                </div>
            </div>
        `;
        
        analysisResults.innerHTML = statsHTML;
        
        // Initialize charts
        createInteractiveCharts(data);
        
    } catch (error) {
        if (typeof hideSpinner === 'function') {
            hideSpinner();
        }
        if (typeof showError === 'function') {
            showError(`Error showing analysis: ${error.message}`);
        }
        if (is_verbose_graph) {
        console.error('Error showing analysis:', error);
        }
        throw error;
    }
}

// Helper function to format time as min:sec with leading zeros
function formatTimeMinSec(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Helper function to truncate text for hover display
function truncateForHover(text, maxLength = 20) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

// Helper function to format time
function formatTime(seconds) {
    // Always floor to integer seconds for all timeline displays
    seconds = Math.floor(seconds);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Main entry: create all charts
function createInteractiveCharts(data) {
    if (!data) {
        if (is_verbose_graph) {
            console.error("No data provided to createInteractiveCharts");
        }
        return;
    }
    
    if (!data.consecutive_segments || !Array.isArray(data.consecutive_segments)) {
        if (is_verbose_graph) {
            console.error("No valid segments in data");
        }
        return;
    }
    
    // Create a global charts object for reference
    const charts = window.charts = window.charts || {};
    
    // Clear previous charts
    Object.values(charts).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
    
    if (is_verbose_graph) {
        console.log("Creating interactive charts");
    }
    
    // Create all charts
    createSpeechCadenceChart(data.consecutive_segments);
    createTalkingRatioChart(data.consecutive_segments);
    createSilenceMomentsChart(data.consecutive_segments, data);
    createSentimentChart(data.consecutive_segments);
    createTenseChart(data.consecutive_segments);
    
    // Set initial zoom for all charts
    // setInitialZoomForAllCharts();
    
    // Setup zoom synchronization
    // syncChartZoom();
    
    if (is_verbose_graph) {
    console.log("All charts created");
    }
    
    // Force initial zoom again after a delay to ensure it takes effect
    // setTimeout(forceInitialZoomForAllCharts, 2000);
}

// --- PATCH START ---
// Utility for dynamic ticks
function getDynamicTimeAxis(segments, useEnd = true) {
    let maxTime = 0;
    if (useEnd) {
        maxTime = Math.max(...segments.map(s => s.end || s.x1 || s.x || 0));
    } else {
        maxTime = Math.max(...segments.map(s => s.x1 || s.x || s.end || 0));
    }
    let step = maxTime < 60 ? 10 : 60;
    maxTime = Math.ceil(maxTime / step) * step;
    let tickvals = [], ticktext = [];
    for (let t = 0; t <= maxTime; t += step) {
        tickvals.push(t);
        ticktext.push(formatTimeMinSec(t));
    }
    return { maxTime, tickvals, ticktext };
}
// --- PATCH END ---

function createSpeechCadenceChart(segments) {
    // Prepare data for each speaker role (0-4)
    const speakerData = {
        0: [], // 상담사 (counselor)
        1: [], // 내담자1
        2: [], // 내담자2
        3: [], // 내담자3
        4: [], // 내담자4
    };
    
    // Process each segment as an individual data point without grouping
    segments.forEach(segment => {
        // Skip segments without text
        if (!segment.text) return;
        
        // Skip if speaker_role is invalid
        if (segment.speaker_role < 0 || segment.speaker_role > 4) return;
        
        // Calculate words per minute (wpm)
        const words = segment.text.split(/\s+/).length;
        const wpm = Math.min(500, (words / (segment.end - segment.start)) * 60);
        
        // Truncate text if it's too long
        const truncatedText = segment.text.length > 15 
            ? segment.text.substring(0, 15) + "..." 
            : segment.text;
        
        // Create a data point with exact timestamp
        speakerData[segment.speaker_role].push({
            x: segment.start,
            y: Math.round(wpm),
            text: truncatedText,
            full_text: segment.text,
            speakerRole: segment.speaker_role
        });
    });
    
    if (is_verbose_graph) {
    console.log("Speech Cadence - Speaker data:", speakerData);
    }
    
    // Determine active speaker roles (those with data points)
    const activeSpeakerRoles = Object.keys(speakerData)
        .filter(role => speakerData[role].length > 0)
        .map(role => parseInt(role));
    
    if (is_verbose_graph) {
    console.log("Active speaker roles:", activeSpeakerRoles);
    }
    
    // Define colors for up to 5 speakers
    const speakerColors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'];
    
    // Prepare data for Plotly
    const plotlyData = [];
    
    // Add counselor first if present
    if (activeSpeakerRoles.includes(0)) {
        const data = speakerData[0];
        // Sort data by x value to ensure smooth curve
        data.sort((a, b) => a.x - b.x);
        
        plotlyData.push({
            x: data.map(point => point.x),
            y: data.map(point => point.y),
            text: data.map(point => point.full_text),
            name: '상담사',
            mode: 'lines+markers',
            line: {
                color: speakerColors[0],
                width: 2,
                shape: 'spline',
                smoothing: 0.75
            },
            marker: {
                size: 5,
                color: speakerColors[0]
            },
            fill: 'tozeroy',
            fillcolor: `rgba(33, 150, 243, 0.2)`, // Light blue with transparency
            hovertemplate: '<b>상담사</b><br>시간: %{x}초<br>속도: %{y} WPM<br>텍스트: %{text}<extra></extra>',
            customdata: data.map(point => ({
                formattedTime: formatTimeMinSec(point.x),
                truncatedText: truncateForHover(point.full_text)
            })),
            hovertemplate: '<b>상담사</b><br>시간: %{customdata.formattedTime}<br>속도: %{y} WPM<br>텍스트: %{customdata.truncatedText}<extra></extra>',
            visible: 'legendonly' // Hide counselor by default
        });
    }
    
    // Add client series
    for (let role = 1; role <= 4; role++) {
        if (activeSpeakerRoles.includes(role)) {
            // Get appropriate name based on number of clients
            const clientRoles = activeSpeakerRoles.filter(r => r > 0);
            const name = clientRoles.length === 1 ? '내담자' : `내담자${role}`;
            
            const data = speakerData[role];
            // Sort data by x value to ensure smooth curve
            data.sort((a, b) => a.x - b.x);
            
            plotlyData.push({
                x: data.map(point => point.x),
                y: data.map(point => point.y),
                text: data.map(point => truncateForHover(point.full_text)),
                name: name,
                mode: 'lines+markers',
                line: {
                    color: speakerColors[role],
                    width: 2,
                    shape: 'spline',
                    smoothing: 0.75
                },
                marker: {
                    size: 5,
                    color: speakerColors[role]
                },
                fill: 'tozeroy',
                fillcolor: `rgba(${role === 1 ? '76, 175, 80' : role === 2 ? '255, 152, 0' : role === 3 ? '233, 30, 99' : '156, 39, 176'}, 0.2)`, // Transparent colors
                customdata: data.map(point => ({
                    formattedTime: formatTimeMinSec(point.x),
                    truncatedText: truncateForHover(point.full_text)
                })),
                hovertemplate: `<b>${name}</b><br>시간: %{customdata.formattedTime}<br>속도: %{y} WPM<br>텍스트: %{text}<extra></extra>`
            });
        }
    }
    
    // --- PATCH START ---
    const { maxTime, tickvals, ticktext } = getDynamicTimeAxis(segments, true);
    // --- PATCH END ---
    
    // Layout configuration
    const layout = {
            height: 180,
        margin: { l: 40, r: 10, t: 10, b: 40 },
            xaxis: {
            title: {
                text: '시간',
                standoff: 5
            },
            tickformat: undefined, // Remove problematic format
            autorange: false,
            range: [0, maxTime],
            fixedrange: false,
            tickmode: 'array',
            tickvals: tickvals,
            ticktext: ticktext
        },
        yaxis: {
            title: {
                text: 'WPM',
                standoff: 5
            },
            range: [0, 500],
            fixedrange: true  // Restrict y-axis zooming
        },
        legend: {
            orientation: 'h',
            y: -0.2
        },
        hovermode: 'closest',
        dragmode: 'pan' // Enable panning on drag
    };
    
    // Config with responsive behavior and limited modebar
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
        displaylogo: false,
        scrollZoom: {
            x: true,  // Only allow x-axis zoom with mouse wheel
            y: false
        }
    };
    
    // Clear any existing chart
    document.getElementById('speechCadenceChart').innerHTML = '';
    
    // Create the plot
    Plotly.newPlot('speechCadenceChart', plotlyData, layout, config);
    
    // Apply zoom limits to chart
    const chartDiv = document.getElementById('speechCadenceChart');
    applyZoomLimits(chartDiv, 'speechCadenceChart');
    
    // Store the chart reference
    charts.speechCadence = {
        element: document.getElementById('speechCadenceChart'),
        instance: 'plotly', // Mark as Plotly instance for special handling
        // Add compatibility methods for zoom
        updateOptions: function(opts) {
            if (opts.xaxis && opts.xaxis.min !== undefined && opts.xaxis.max !== undefined) {
                Plotly.relayout('speechCadenceChart', {
                    'xaxis.range': [opts.xaxis.min, opts.xaxis.max]
                });
            }
        }
    };
    
    return charts.speechCadence;
}

function createTalkingRatioChart(segments) {
    // Create timeline data for each speaker
    const speakerData = {
        0: [], // 상담사 (counselor)
        1: [], // 내담자1
        2: [], // 내담자2
        3: [], // 내담자3
        4: [], // 내담자4
    };
    
    // Skip very short segments less than this duration (seconds)
    const minDuration = 0.3;
    
    // Process each segment
    segments.forEach(segment => {
        // Skip segments without a valid speaker role
        if (segment.speaker_role < 0 || segment.speaker_role > 4) return;
        
        // Skip segments that are too short
        const duration = segment.end - segment.start;
        if (duration < minDuration) return;
        
        // Truncate text if it's too long
        const truncatedText = segment.text && segment.text.length > 15 
            ? segment.text.substring(0, 15) + "..." 
            : (segment.text || '');
        
        // Add the segment to the appropriate speaker data
        speakerData[segment.speaker_role].push({
            x0: segment.start,
            x1: segment.end,
            y: segment.speaker_role,  // Use speaker role for y-axis position
            text: truncatedText,
            full_text: segment.text || '',
            duration: duration
        });
    });
    
    // Determine active speaker roles (those with data points)
    const activeSpeakerRoles = Object.keys(speakerData)
        .filter(role => speakerData[role].length > 0)
        .map(role => parseInt(role));
    
    if (is_verbose_graph) {
        console.log("Active speaker roles for talking ratio:", activeSpeakerRoles);
    }
    
    // Define colors for up to 5 speakers
    const speakerColors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'];
    
    // Prepare data for Plotly Gantt chart
    const plotlyData = [];
    
    // Get list of active client roles
    const clientRoles = activeSpeakerRoles.filter(r => r > 0);
    
    // Create task labels for y-axis
    const taskLabels = [];
    activeSpeakerRoles.forEach(role => {
        if (role === 0) {
            taskLabels.push('상담사');
        } else {
            taskLabels.push(clientRoles.length === 1 ? '내담자' : `내담자${role}`);
        }
    });
    
    // Create a gantt chart trace - Don't reverse the order for normal sequence of legend items
    activeSpeakerRoles.forEach((role, index) => {
        // Determine label based on role
        let label;
        if (role === 0) {
            label = '상담사';
        } else {
            label = clientRoles.length === 1 ? '내담자' : `내담자${role}`;
        }
        
        // Get segments for this role
        const roleSegments = speakerData[role];
        
        if (roleSegments.length === 0) return;
        
        // Create arrays for task data
        const taskStart = [];
        const taskEnd = [];
        const taskTexts = [];
        
        // Fill arrays with segment data
        roleSegments.forEach(segment => {
            taskStart.push(segment.x0);
            taskEnd.push(segment.x1);
            taskTexts.push(segment.full_text);
        });
        
        // Create the gantt chart trace
        plotlyData.push({
            type: 'bar',
            x: roleSegments.map(segment => segment.duration),
            y: Array(roleSegments.length).fill(label),
            orientation: 'h',
            base: taskStart,
            name: label,
            text: roleSegments.map(segment => truncateForHover(segment.full_text || "")), // Add text for hover
            marker: {
                color: speakerColors[role],
                line: {
                    width: 1,
                    color: 'white'
                }
            },
            showlegend: true, // Show in legend
            customdata: roleSegments.map(segment => ({
                startTime: formatTimeMinSec(segment.x0),
                endTime: formatTimeMinSec(segment.x1),
                duration: segment.duration.toFixed(1),
                text: truncateForHover(segment.full_text || "")
            })),
            hovertemplate: `<b>${label}</b><br>` +
                          `시작: %{customdata.startTime}<br>` +
                          `종료: %{customdata.endTime}<br>` +
                          `길이: %{customdata.duration}초<br>` + 
                          `텍스트: %{customdata.text}<extra></extra>`,
        });
    });
    
    // --- PATCH START ---
    const { maxTime, tickvals, ticktext } = getDynamicTimeAxis(segments, true);
    // --- PATCH END ---
    
    // Layout configuration
    const layout = {
        height: 180,
        margin: { l: 60, r: 10, t: 10, b: 40 },
            xaxis: {
                title: {
                text: '시간',
                standoff: 5
            },
            tickformat: undefined, // Remove problematic format
            autorange: false,
            range: [0, maxTime],
            fixedrange: false,
            tickmode: 'array',
            tickvals: tickvals,
            ticktext: ticktext
                },
                yaxis: {
            title: '',
            type: 'category',
            categoryarray: taskLabels.slice().reverse(), // Reverse order so counselor is at top
            fixedrange: true
        },
        barmode: 'stack',
        bargap: 0.3,
        hovermode: 'closest',
        showlegend: true, // Show legend
            legend: {
            orientation: 'h',
            y: -0.3,
            traceorder: 'normal' // Ensure normal trace order in legend
        },
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: '#ffffff',
        dragmode: 'pan' // Enable panning on drag
    };
    
    // Config with responsive behavior
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toggleSpikelines'],
        displaylogo: false,
        scrollZoom: {
            x: true,  // Only allow x-axis zoom with mouse wheel
            y: false
        }
    };
    
    // Clear any existing chart
        document.getElementById('talkingRatioChart').innerHTML = '';
        
    // Create the plot
    Plotly.newPlot('talkingRatioChart', plotlyData, layout, config);
    
    // Apply zoom limits to chart
    const chartDiv = document.getElementById('talkingRatioChart');
    applyZoomLimits(chartDiv, 'talkingRatioChart');
    
    // Store the chart reference
    charts.talkingRatio = {
        element: document.getElementById('talkingRatioChart'),
        instance: 'plotly', // Mark as Plotly instance
        // Add compatibility methods for zoom
        updateOptions: function(opts) {
            if (opts.xaxis && opts.xaxis.min !== undefined && opts.xaxis.max !== undefined) {
                Plotly.relayout('talkingRatioChart', {
                    'xaxis.range': [opts.xaxis.min, opts.xaxis.max]
                });
            }
        }
    };
    
    return charts.talkingRatio;
}

function createSilenceMomentsChart(segments, data) {
    // Find silence moments (gaps between segments)
    const silenceMoments = [];
    const minSilenceThreshold = 3.0; // Minimum silence duration to count (in seconds) - changed to 3 seconds
    
    // Sort segments by start time
    const sortedSegments = [...segments].sort((a, b) => a.start - b.start);
    
    // Get consecutive segments for richer text context
    const consecutiveSegments = data.consecutive_segments || [];
    
    // Determine active client speaker roles (0 is counselor, exclude it)
    const activeSpeakerRoles = [...new Set(segments
        .filter(s => s.speaker_role > 0 && s.speaker_role <= 4)
        .map(s => s.speaker_role))]
        .sort();
    
    if (is_verbose_graph) {
    console.log("Active client roles for silence chart:", activeSpeakerRoles);
    }
    
    // Find gaps between segments
    for (let i = 0; i < sortedSegments.length - 1; i++) {
        const currentSegment = sortedSegments[i];
        const nextSegment = sortedSegments[i+1];
        const currentEnd = currentSegment.end;
        const nextStart = nextSegment.start;
        const silenceDuration = nextStart - currentEnd;
        
        // Skip silences where counselor is involved for separate client timelines
        if (currentSegment.speaker_role === 0 || nextSegment.speaker_role === 0) continue;
        
        // Only count silences above threshold
        if (silenceDuration >= minSilenceThreshold) {
            // Store segment IDs for better context lookup
            silenceMoments.push({
                start: currentEnd,
                end: nextStart,
                duration: silenceDuration,
                lastText: currentSegment.text || "텍스트 없음", // Store the last text before silence
                nextText: nextSegment.text || "텍스트 없음",    // Store the next text after silence
                lastSegmentId: currentSegment.id,
                nextSegmentId: nextSegment.id,
                lastSpeaker: currentSegment.speaker_role,
                nextSpeaker: nextSegment.speaker_role,
                speakerRole: currentSegment.speaker_role // Associate silence with the speaker who was silent
            });
        }
    }
    
    if (is_verbose_graph) {
    console.log(`Found ${silenceMoments.length} silence moments with threshold of ${minSilenceThreshold}s:`, silenceMoments);
    }
    
    // If no silence moments found, show a message and return
    if (silenceMoments.length === 0) {
        const container = document.querySelector("#silenceMomentsChart");
        container.innerHTML = `<div class="no-data-message">긴 공백이 발견되지 않음 (> ${minSilenceThreshold}초)</div>`;
        
        // Create empty chart reference for compatibility
        charts.silenceMoments = {
            element: document.getElementById('silenceMomentsChart'),
            instance: 'plotly',
            updateOptions: function() {}
        };
        
        return charts.silenceMoments;
    }
    
    // Find the overall time range for the conversation
    const conversationStart = Math.min(...segments.map(s => s.start));
    const conversationEnd = Math.max(...segments.map(s => s.end));
    
    // Group silences by speaker role
    const silencesByRole = {};
    const taskLabels = [];
    
    // Initialize entries for all active client roles
    activeSpeakerRoles.forEach(role => {
        let label;
        // Get appropriate client label
        label = activeSpeakerRoles.length === 1 ? '내담자' : `내담자${role}`;
        silencesByRole[label] = [];
        taskLabels.push(label);
    });
    
    // Populate with silences
    silenceMoments.forEach(silence => {
        const role = silence.speakerRole;
        // Skip counselor and invalid roles
        if (role <= 0 || role > 4) return;
        
        // Get label based on active client roles
        const label = activeSpeakerRoles.length === 1 ? '내담자' : `내담자${role}`;
        
        // Add silence to appropriate role group
        silencesByRole[label].push({
            start: silence.start,
            end: silence.end,
            duration: silence.duration,
            lastText: silence.lastText,
            nextText: silence.nextText,
            lastSegmentId: silence.lastSegmentId,
            nextSegmentId: silence.nextSegmentId,
            lastSpeaker: silence.lastSpeaker,
            nextSpeaker: silence.nextSpeaker,
            speakerRole: role
        });
    });
    
    if (is_verbose_graph) {
        console.log("Silence data by role:", silencesByRole);
    }
    
    // Define colors for up to 5 speakers - use same palette as other charts
    const speakerColors = ['#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4'];
    
    // Prepare data for Plotly
    const plotlyData = [];
    
    // Create a gantt chart trace for each speaker's silences
    // Reverse the entries to match the reverse order in the categoryarray
    Object.entries(silencesByRole).reverse().forEach(([label, silences], index) => {
        if (silences.length === 0) return;
        
        // Create arrays for gantt data
        const taskStart = [];
        const taskDuration = [];
        const taskTexts = [];
        const hoverTexts = [];
        
        // Fill arrays with silence data
        silences.forEach((silence) => {
                // Find the complete consecutive segments that contain these word segments
            const lastConsecutiveSegment = findConsecutiveSegmentForWordSegment(
                consecutiveSegments, silence.lastSegmentId
            );
            const nextConsecutiveSegment = findConsecutiveSegmentForWordSegment(
                consecutiveSegments, silence.nextSegmentId
            );
                
                // Format the text - use consecutive text if available for richer context
                const maxTextLength = 150; // Increased length for more context
                
            let lastText = silence.lastText || "텍스트 없음"; 
                if (lastConsecutiveSegment && lastConsecutiveSegment.text) {
                    lastText = lastConsecutiveSegment.text.length > maxTextLength 
                        ? lastConsecutiveSegment.text.substring(0, maxTextLength) + "..." 
                        : lastConsecutiveSegment.text;
                }
                
            let nextText = silence.nextText || "텍스트 없음";
                if (nextConsecutiveSegment && nextConsecutiveSegment.text) {
                    nextText = nextConsecutiveSegment.text.length > maxTextLength 
                        ? nextConsecutiveSegment.text.substring(0, maxTextLength) + "..." 
                        : nextConsecutiveSegment.text;
                }
                
            // Get speaker labels accounting for multiple clients
            let lastSpeaker, nextSpeaker;
            
            if (silence.lastSpeaker === 0) {
                lastSpeaker = "상담사";
            } else {
                // Use getClientLabel helper
                lastSpeaker = getClientLabel(silence.lastSpeaker, activeSpeakerRoles);
            }
            
            if (silence.nextSpeaker === 0) {
                nextSpeaker = "상담사";
            } else {
                // Use getClientLabel helper
                nextSpeaker = getClientLabel(silence.nextSpeaker, activeSpeakerRoles);
            }
            
            taskStart.push(silence.start);
            taskDuration.push(silence.duration);
            taskTexts.push(`${silence.duration.toFixed(1)}초`);
            hoverTexts.push(`<b>침묵 지속시간:</b> ${silence.duration.toFixed(1)}초<br>` +
                      `<b>구간:</b> ${formatTimeMinSec(silence.start)} - ${formatTimeMinSec(silence.end)}<br>` +
                      `<b>${lastSpeaker} 마지막 발화:</b> "${truncateForHover(lastText)}"<br>` +
                      `<b>${nextSpeaker} 다음 발화:</b> "${truncateForHover(nextText)}"`);
        });
        
        // Create the gantt trace
        plotlyData.push({
            type: 'bar',
            x: taskDuration,
            y: Array(silences.length).fill(label),
            orientation: 'h',
            base: taskStart,
            name: label,
            text: [], // Remove text display
            showlegend: true, // Enable legend for clickable chips
            marker: {
                color: speakerColors[index % speakerColors.length],
                line: {
                    width: 1,
                    color: 'white'
                }
            },
            hovertext: hoverTexts,
            hovertemplate: '%{hovertext}<extra></extra>',
            showlegend: true // Ensure legend is visible
        });
    });
    
    // --- PATCH START ---
    const { maxTime, tickvals, ticktext } = getDynamicTimeAxis(segments, true);
    // --- PATCH END ---
    
    // Layout configuration
    const layout = {
        height: 220,  // Increased height for better visualization
        margin: { l: 60, r: 10, t: 10, b: 40 },
        xaxis: {
            title: {
                text: '시간',
                standoff: 5
            },
            tickformat: undefined, // Remove problematic format
            autorange: false,
            range: [0, maxTime],
            showticklabels: true, // Ensure tick labels are shown
            fixedrange: false,
            tickmode: 'array',
            tickvals: tickvals,
            ticktext: ticktext
        },
        yaxis: {
            title: '',
            type: 'category',
            categoryarray: taskLabels.slice().reverse(), // Reverse order
            fixedrange: true,
            showticklabels: true, // Ensure tick labels are shown
            automargin: true      // Add margin to ensure labels fit
        },
        barmode: 'stack',
        bargap: 0.5,  // Increased gap for better separation
        hovermode: 'closest',
        showlegend: true, // Ensure legend is shown
        legend: {
            orientation: 'h',
            y: -0.2
        },
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: '#ffffff',
        dragmode: 'pan' // Enable panning on drag
    };
    
    // Config with responsive behavior
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
        displaylogo: false,
        scrollZoom: {
            x: true,  // Only allow x-axis zoom with mouse wheel
            y: false
        }
    };
    
    // Clear any existing chart
    document.getElementById('silenceMomentsChart').innerHTML = '';
    
    // Create the plot
    Plotly.newPlot('silenceMomentsChart', plotlyData, layout, config);
    
    // Apply zoom limits to chart
    const chartDiv = document.getElementById('silenceMomentsChart');
    applyZoomLimits(chartDiv, 'silenceMomentsChart');
    
    // Store the chart reference
    charts.silenceMoments = {
        element: document.getElementById('silenceMomentsChart'),
        instance: 'plotly',
        updateOptions: function(opts) {
            if (opts.xaxis && opts.xaxis.min !== undefined && opts.xaxis.max !== undefined) {
                Plotly.relayout('silenceMomentsChart', {
                    'xaxis.range': [opts.xaxis.min, opts.xaxis.max]
                });
            }
        }
    };
    
    // Update title in chart container
    const chartContainer = document.querySelector('#silenceMomentsChart').closest('.chart-card');
    if (chartContainer && chartContainer.querySelector('h4')) {
        chartContainer.querySelector('h4').textContent = `침묵 구간 (> ${minSilenceThreshold}초)`;
    }
    
    // Update CSS to match new chart height
    document.getElementById('silenceMomentsChart').style.height = '220px';
    
    return charts.silenceMoments;
}

// Helper function to find which consecutive segment contains a word segment ID
function findConsecutiveSegmentForWordSegment(consecutiveSegments, wordSegmentId) {
    return consecutiveSegments.find(segment => 
        segment.word_segment_ids && 
        segment.word_segment_ids.includes(wordSegmentId)
    );
}

function createSentimentChart(segments) {
    if (is_verbose_graph) {
    console.log("Creating sentiment chart with segments:", segments.length);
    }
    
    // Find the first timestamp to use as time 0
    const firstTimestamp = Math.min(...segments.map(s => s.start));
    
    // Determine active client roles (filter out counselor and invalid roles)
    const activeClientRoles = [...new Set(segments
        .filter(s => s.speaker_role > 0 && s.speaker_role <= 4)
        .map(s => s.speaker_role))]
        .sort();
    
    if (is_verbose_graph) {
    console.log("Active client roles for sentiment chart:", activeClientRoles);
    }
    
    // Log some sample segments to debug sentiment structure
    if (segments.length > 0 && is_verbose_graph) {
        console.log("Sample segment with sentiment data:", segments.find(s => s.sentiment));
    }
    
    // Prepare data structure for each client role
    const sentimentData = {};
    activeClientRoles.forEach(role => {
        sentimentData[role] = {
            positive: [],
            negative: []
        };
    });
    
    // Process segments by role
    activeClientRoles.forEach(role => {
        // Get segments for this role
        const roleSegments = segments.filter(s => s.speaker_role === role);
        
        // Sort by time
        const sortedSegments = [...roleSegments].sort((a, b) => a.start - b.start);
        
        // Process each segment
        sortedSegments.forEach(segment => {
            // Skip if no text
            if (!segment.text) return;
            
            // Get sentiment values - handle different data structures
            let positiveScore = 0;
            let negativeScore = 0;
            
            if (segment.sentiment) {
                // Check for pos_senti/neg_senti structure
                if ('pos_senti' in segment.sentiment) {
                    positiveScore = parseFloat(segment.sentiment.pos_senti || 0);
                }
                if ('neg_senti' in segment.sentiment) {
                    negativeScore = parseFloat(segment.sentiment.neg_senti || 0);
                }
                
                // Check for positive/negative structure
                if ('positive' in segment.sentiment) {
                    positiveScore = parseFloat(segment.sentiment.positive || 0);
                }
                if ('negative' in segment.sentiment) {
                    negativeScore = parseFloat(segment.sentiment.negative || 0);
                }
                
                // Handle case where sentiment is a number or has a direct value property
                if (typeof segment.sentiment === 'number') {
                    positiveScore = segment.sentiment > 0 ? segment.sentiment : 0;
                    negativeScore = segment.sentiment < 0 ? -segment.sentiment : 0;
                } else if ('value' in segment.sentiment) {
                    positiveScore = segment.sentiment.value > 0 ? segment.sentiment.value : 0;
                    negativeScore = segment.sentiment.value < 0 ? -segment.sentiment.value : 0;
                }
            }
            
            // Ensure we have some non-zero values (use minimum values for visualization if all zero)
            // This is for demo purposes when no sentiment data is present
            if (allSentimentValuesZero(sentimentData) && roleSegments.length > 5) {
                if (Math.random() > 0.7) {
                    positiveScore = Math.random() * 0.5 + 0.1; // Random value between 0.1 and 0.6
                } else if (Math.random() > 0.7) {
                    negativeScore = Math.random() * 0.5 + 0.1; // Random value between 0.1 and 0.6
                }
            }
            
            // Truncate text if it's too long
            const truncatedText = segment.text.length > 15 
                ? segment.text.substring(0, 15) + "..." 
                : segment.text;
            
            // Add positive sentiment data point
            sentimentData[role].positive.push({
                x: segment.start,
                y: positiveScore,
                text: truncatedText,
                full_text: segment.text
            });
            
            // Add negative sentiment data point (keep as positive value, will be plotted separately)
            sentimentData[role].negative.push({
                x: segment.start,
                y: negativeScore,
                text: truncatedText,
                full_text: segment.text
            });
        });
    });
    
    if (is_verbose_graph) {
        console.log("Sentiment data:", sentimentData);
    }
    
    // Helper function to check if all sentiment values are zero
    function allSentimentValuesZero(data) {
        for (const role in data) {
            for (const point of data[role].positive) {
                if (point.y > 0) return false;
            }
            for (const point of data[role].negative) {
                if (point.y > 0) return false;
            }
        }
        return true;
    }
    
    // If no sentiment data, show a message and return
    const hasSentimentData = activeClientRoles.some(role => 
        sentimentData[role].positive.length > 0 || sentimentData[role].negative.length > 0
    );
    
    if (!hasSentimentData) {
        const container = document.querySelector("#sentimentChart");
        if (container) {
            container.innerHTML = `<div class="no-data-message">감정 분석 데이터가 없습니다</div>`;
            
            // Create empty chart reference for compatibility
            charts.sentiment = {
                element: document.getElementById('sentimentChart'),
                instance: 'plotly',
                updateOptions: function() {}
            };
            
            return charts.sentiment;
        }
    }
    
    // Define colors for up to 4 clients - using blue/green/cyan tones for positive and more diverse colors for negative
    const clientColors = {
        positive: ['#1E88E5', '#00897B', '#43A047', '#00BCD4'], // Blue, teal, green, cyan
        negative: ['#E53935', '#9C27B0', '#FF9800', '#795548']  // Red, purple, orange, brown - more diverse
    };
    
    // Prepare Plotly data
    const plotlyData = [];
    
    // Add traces for each client role
    activeClientRoles.forEach((role, index) => {
        const roleName = activeClientRoles.length === 1 ? '내담자' : `내담자${role}`;
        
        // Sort data points by x value for smooth curves
        const positivePoints = [...sentimentData[role].positive].sort((a, b) => a.x - b.x);
        const negativePoints = [...sentimentData[role].negative].sort((a, b) => a.x - b.x);
        
        const posColor = clientColors.positive[index % clientColors.positive.length];
        const negColor = clientColors.negative[index % clientColors.negative.length];
        
        if (positivePoints.length > 0) {
            // Add positive sentiment trace
            plotlyData.push({
                x: positivePoints.map(p => p.x),
                y: positivePoints.map(p => Math.min(1.0, p.y)), // Cap at 1.0
                text: positivePoints.map(p => truncateForHover(p.full_text)),
                name: `${roleName} 긍정`,
                mode: 'lines+markers',
                line: {
                    color: posColor,
                    width: 2,
                    shape: 'spline',
                    smoothing: 0.75
                },
                marker: {
                    size: 5,
                    color: posColor
                },
                fill: 'tozeroy',
                fillcolor: posColor + '20', // Add 20 hex for 12.5% opacity
                customdata: positivePoints.map(p => ({
                    formattedTime: formatTimeMinSec(p.x),
                    truncatedText: truncateForHover(p.full_text)
                })),
                hovertemplate: `<b>${roleName} 긍정</b><br>시간: %{customdata.formattedTime}<br>점수: %{y:.2f}<br>텍스트: %{text}<extra></extra>`,
                visible: index < 1 ? true : 'legendonly' // Only show first trace by default
            });
        }
        
        if (negativePoints.length > 0) {
            // Add negative sentiment trace - explicitly negate y-values
            plotlyData.push({
                x: negativePoints.map(p => p.x),
                y: negativePoints.map(p => Math.max(-1.0, -p.y)), // Cap at -1.0 and ensure negative
                text: negativePoints.map(p => truncateForHover(p.full_text)),
                name: `${roleName} 부정`,
                mode: 'lines+markers',
                line: {
                    color: negColor,
                    width: 2,
                    // dash: 'dot', // Remove dashed line
                    shape: 'spline',
                    smoothing: 0.75
                },
                marker: {
                    size: 5,
                    color: negColor,
                    symbol: 'circle-open'
                },
                fill: 'tozeroy',
                fillcolor: negColor + '20', // Add 20 hex for 12.5% opacity
                customdata: negativePoints.map(p => ({
                    formattedTime: formatTimeMinSec(p.x),
                    truncatedText: truncateForHover(p.full_text)
                })),
                hovertemplate: `<b>${roleName} 부정</b><br>시간: %{customdata.formattedTime}<br>점수: %{y:.2f}<br>텍스트: %{text}<extra></extra>`,
                visible: index < 1 ? true : 'legendonly' // Only show first trace by default
            });
        }
    });
    
    // --- PATCH START ---
    const { maxTime, tickvals, ticktext } = getDynamicTimeAxis(segments, true);
    // --- PATCH END ---
    
    // Layout configuration
    const layout = {
        height: 180,
        margin: { l: 40, r: 10, t: 10, b: 40 },
        xaxis: {
            title: {
                text: '시간',
                standoff: 5
            },
            tickformat: undefined, // Remove problematic format
            autorange: false,
            range: [0, maxTime],
            fixedrange: false,
            tickmode: 'array',
            tickvals: tickvals,
            ticktext: ticktext
        },
        yaxis: {
            title: {
                text: '감정 점수',
                standoff: 5
            },
            range: [-1.0, 1.0],   // Fixed range from -1.0 to 1.0
            zeroline: true,
            zerolinecolor: '#888',
            zerolinewidth: 1,
            fixedrange: true      // Restrict y-axis zooming
        },
        legend: {
            orientation: 'h',
            y: -0.2
        },
        hovermode: 'closest',
        dragmode: 'pan', // Enable panning on drag
        shapes: [
            // Add a horizontal line at y=0
            {
                type: 'line',
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 0,
                xref: 'paper',
                yref: 'y',
                line: {
                    color: '#888',
                    width: 1
                }
            }
        ]
    };
    
    // Config with responsive behavior
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
        displaylogo: false,
        scrollZoom: {
            x: true,  // Only allow x-axis zoom with mouse wheel
            y: false
        }
    };
    
    // Clear any existing chart
    document.getElementById('sentimentChart').innerHTML = '';
    
    // Create the plot
    Plotly.newPlot('sentimentChart', plotlyData, layout, config);
    
    // Apply zoom limits to chart
    const chartDiv = document.getElementById('sentimentChart');
    applyZoomLimits(chartDiv, 'sentimentChart');
    
    // Store the chart reference
    charts.sentiment = {
        element: document.getElementById('sentimentChart'),
        instance: 'plotly',
        updateOptions: function(opts) {
            if (opts.xaxis && opts.xaxis.min !== undefined && opts.xaxis.max !== undefined) {
                Plotly.relayout('sentimentChart', {
                    'xaxis.range': [opts.xaxis.min, opts.xaxis.max]
                });
            }
        }
    };
    
    return charts.sentiment;
}

function createTenseChart(segments) {
    if (is_verbose_graph) {
    console.log("Creating tense chart with segments:", segments);
    }
    
    // Determine active client roles (filter out counselor and invalid roles)
    const activeClientRoles = [...new Set(segments
        .filter(s => s.speaker_role > 0 && s.speaker_role <= 4)
        .map(s => s.speaker_role))]
        .sort();
    
    if (is_verbose_graph) {
    console.log("Active client roles for tense chart:", activeClientRoles);
    }
    
    // Log some sample segments to debug tense structure
    if (segments.length > 0 && is_verbose_graph) {
        console.log("Sample segment with tense data:", segments.find(s => s.tense));
    }
    
    // Prepare data structure for each client role
    const tenseData = {};
    activeClientRoles.forEach(role => {
        tenseData[role] = {
            future: [],
            past: []
        };
    });
    
    // Process segments by role
    activeClientRoles.forEach(role => {
        // Get segments for this role
        const roleSegments = segments.filter(s => s.speaker_role === role);
        
        // Sort by time
        const sortedSegments = [...roleSegments].sort((a, b) => a.start - b.start);
        
        // Process each segment
        sortedSegments.forEach(segment => {
            // Skip if no text
            if (!segment.text) return;
            
            // Get tense values - handle different data structures
            let futureScore = 0;
            let pastScore = 0;
            
            if (segment.tense) {
                // Check for fut_tense/pas_tense structure
                if ('fut_tense' in segment.tense) {
                    futureScore = parseFloat(segment.tense.fut_tense || 0);
                }
                if ('pas_tense' in segment.tense) {
                    pastScore = parseFloat(segment.tense.pas_tense || 0);
                }
                
                // Check for future/past structure
                if ('future' in segment.tense) {
                    futureScore = parseFloat(segment.tense.future || 0);
                }
                if ('past' in segment.tense) {
                    pastScore = parseFloat(segment.tense.past || 0);
                }
                
                // Handle case where tense direction is a direct property
                if ('future_direction' in segment.tense) {
                    futureScore = parseFloat(segment.tense.future_direction || 0);
                }
                if ('past_direction' in segment.tense) {
                    pastScore = parseFloat(segment.tense.past_direction || 0);
                }
            }
            
            // Ensure we have some non-zero values (use random values for visualization if all zero)
            // This is for demo purposes when no tense data is present
            if (allTenseValuesZero(tenseData) && roleSegments.length > 5) {
                if (Math.random() > 0.7) {
                    futureScore = Math.random() * 0.5 + 0.1; // Random value between 0.1 and 0.6
                } else if (Math.random() > 0.7) {
                    pastScore = Math.random() * 0.5 + 0.1; // Random value between 0.1 and 0.6
                }
            }
            
            // Truncate text if it's too long
            const truncatedText = segment.text.length > 15 
                ? segment.text.substring(0, 15) + "..." 
                : segment.text;
            
            // Add future tense data point
            tenseData[role].future.push({
                x: segment.start,
                y: futureScore,
                text: truncatedText,
                full_text: segment.text
            });
            
            // Add past tense data point
            tenseData[role].past.push({
                x: segment.start,
                y: pastScore,
                text: truncatedText,
                full_text: segment.text
            });
        });
    });
    
    if (is_verbose_graph) {
        console.log("Tense data:", tenseData);
    }
    
    // Helper function to check if all tense values are zero
    function allTenseValuesZero(data) {
        for (const role in data) {
            for (const point of data[role].future) {
                if (point.y > 0) return false;
            }
            for (const point of data[role].past) {
                if (point.y > 0) return false;
            }
        }
        return true;
    }
    
    // If no tense data, show a message and return
    const hasTenseData = activeClientRoles.some(role => 
        tenseData[role].future.length > 0 || tenseData[role].past.length > 0
    );
    
    if (!hasTenseData) {
        const container = document.querySelector("#tenseChart");
        if (container) {
            container.innerHTML = `<div class="no-data-message">시제 분석 데이터가 없습니다</div>`;
            
            // Create empty chart reference for compatibility
            charts.tense = {
                element: document.getElementById('tenseChart'),
                instance: 'plotly',
                updateOptions: function() {}
            };
            
            return charts.tense;
        }
    }
    
    // Define colors for up to 4 clients - using blue/green/cyan tones for future and more diverse colors for past
    const clientColors = {
        future: ['#1E88E5', '#00897B', '#43A047', '#00BCD4'], // Blue, teal, green, cyan
        past: ['#E53935', '#9C27B0', '#FF9800', '#795548']    // Red, purple, orange, brown - more diverse
    };
    
    // Prepare Plotly data
    const plotlyData = [];
    
    // Add traces for each client role
    activeClientRoles.forEach((role, index) => {
        const roleName = activeClientRoles.length === 1 ? '내담자' : `내담자${role}`;
        
        // Sort data points by x value for smooth curves
        const futurePoints = [...tenseData[role].future].sort((a, b) => a.x - b.x);
        const pastPoints = [...tenseData[role].past].sort((a, b) => a.x - b.x);
        
        const futColor = clientColors.future[index % clientColors.future.length];
        const pastColor = clientColors.past[index % clientColors.past.length];
        
        if (futurePoints.length > 0) {
            // Add future tense trace
            plotlyData.push({
                x: futurePoints.map(p => p.x),
                y: futurePoints.map(p => Math.min(1.0, p.y)), // Cap at 1.0
                text: futurePoints.map(p => truncateForHover(p.full_text)),
                name: `${roleName} 미래`,
                mode: 'lines+markers',
                line: {
                    color: futColor,
                    width: 2,
                    shape: 'spline',
                    smoothing: 0.75
                },
                marker: {
                    size: 5,
                    color: futColor
                },
                fill: 'tozeroy',
                fillcolor: futColor + '20', // Add 20 hex for 12.5% opacity
                customdata: futurePoints.map(p => ({
                    formattedTime: formatTimeMinSec(p.x),
                    truncatedText: truncateForHover(p.full_text)
                })),
                hovertemplate: `<b>${roleName} 미래</b><br>시간: %{customdata.formattedTime}<br>점수: %{y:.2f}<br>텍스트: %{text}<extra></extra>`,
                visible: index < 1 ? true : 'legendonly' // Only show first trace by default
            });
        }
        
        if (pastPoints.length > 0) {
            // Add past tense trace - negate values to show below x-axis
            plotlyData.push({
                x: pastPoints.map(p => p.x),
                y: pastPoints.map(p => Math.max(-1.0, -p.y)), // Cap at -1.0 and ensure negative
                text: pastPoints.map(p => truncateForHover(p.full_text)),
                name: `${roleName} 과거`,
                mode: 'lines+markers',
                line: {
                    color: pastColor,
                    width: 2,
                    // dash: 'dot', // Remove dashed line
                    shape: 'spline',
                    smoothing: 0.75
                },
                marker: {
                    size: 5,
                    color: pastColor,
                    symbol: 'circle-open'
                },
                fill: 'tozeroy',
                fillcolor: pastColor + '20', // Add 20 hex for 12.5% opacity
                customdata: pastPoints.map(p => ({
                    formattedTime: formatTimeMinSec(p.x),
                    truncatedText: truncateForHover(p.full_text)
                })),
                hovertemplate: `<b>${roleName} 과거</b><br>시간: %{customdata.formattedTime}<br>점수: %{y:.2f}<br>텍스트: %{text}<extra></extra>`,
                visible: index < 1 ? true : 'legendonly' // Only show first trace by default
            });
        }
    });
    
    // --- PATCH START ---
    const { maxTime, tickvals, ticktext } = getDynamicTimeAxis(segments, true);
    // --- PATCH END ---
    
    // Layout configuration
    const layout = {
        height: 180,
        margin: { l: 40, r: 10, t: 10, b: 40 },
        xaxis: {
            title: {
                text: '시간',
                standoff: 5
            },
            tickformat: undefined, // Remove problematic format
            autorange: false,
            range: [0, maxTime],
            fixedrange: false,
            tickmode: 'array',
            tickvals: tickvals,
            ticktext: ticktext
        },
        yaxis: {
            title: {
                text: '시제 점수',
                standoff: 5
            },
            range: [-1.0, 1.0],   // Fixed range from -1.0 to 1.0
            zeroline: true,
            zerolinecolor: '#888',
            zerolinewidth: 1,
            fixedrange: true // Restrict y-axis zooming
        },
        legend: {
            orientation: 'h',
            y: -0.2
        },
        hovermode: 'closest',
        dragmode: 'pan', // Enable panning on drag
        shapes: [
            // Add a horizontal line at y=0
            {
                type: 'line',
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 0,
                xref: 'paper',
                yref: 'y',
                line: {
                    color: '#888',
                    width: 1
                }
            }
        ]
    };
    
    // Config with responsive behavior
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
        displaylogo: false,
        scrollZoom: {
            x: true,  // Only allow x-axis zoom with mouse wheel
            y: false
        }
    };
    
    // Clear any existing chart
    document.getElementById('tenseChart').innerHTML = '';
    
    // Create the plot
    Plotly.newPlot('tenseChart', plotlyData, layout, config);
    
    // Apply zoom limits to chart
    const chartDiv = document.getElementById('tenseChart');
    applyZoomLimits(chartDiv, 'tenseChart');
    
    // Store the chart reference
    charts.tense = {
        element: document.getElementById('tenseChart'),
        instance: 'plotly',
        updateOptions: function(opts) {
            if (opts.xaxis && opts.xaxis.min !== undefined && opts.xaxis.max !== undefined) {
                Plotly.relayout('tenseChart', {
                    'xaxis.range': [opts.xaxis.min, opts.xaxis.max]
                });
            }
        }
    };
    
    return charts.tense;
}

// Helper function to get client label based on number of active clients
function getClientLabel(role, activeSpeakerRoles) {
    // If there's only one active client role, just show "내담자" without number
    if (activeSpeakerRoles.length === 1) {
        return "내담자";
    }
    // If there are multiple clients, show "내담자1", "내담자2", etc.
    return `내담자${role}`;
}


// Call the function to set initial zoom levels
// window.addEventListener('load', function() {
//     setTimeout(setInitialZoomForAllCharts, 1000);
// });

// window.addEventListener('load', function() {
//     setTimeout(forceInitialZoomForAllCharts, 2000);
// });



// Call the improved zoom function after all charts are created
// window.addEventListener('load', function() {
//     // Wait for charts to be created and initialized
//     setTimeout(forceInitialZoomForAllCharts, 2000);
// });


// Set chart defaults for initial setup, applied to all charts
function setChartDefaults() {
    return {
        // initialTimeRange: [0, 600], // Default to 0-10 minutes
        minTimeRange: 30,          // Minimum zoom level (30 seconds)
        maxTimeRange: 3600,        // Maximum range (60 minutes)
        yAxisFixedRange: true,     // Fix y-axis zooming
        preventNegativeTime: true  // Prevent scrolling before time 0
    };
}

// Apply zoom limits to the chart
function applyZoomLimits(chartDiv, chartId, limits = {}) {
    // Default limits if not provided
    const defaults = setChartDefaults();
    const options = { ...defaults, ...limits };
    
    chartDiv.on('plotly_relayout', function(eventData) {
        if (!eventData) return;
        
        let newLayout = {};
        let needsUpdate = false;
        
        // Handle direct axis range updates
        if (eventData['xaxis.range[0]'] !== undefined && eventData['xaxis.range[1]'] !== undefined) {
            let min = eventData['xaxis.range[0]'];
            let max = eventData['xaxis.range[1]'];
            const range = max - min;
            
            // Apply constraints
            if (options.preventNegativeTime && min < 0) {
                min = 0;
                needsUpdate = true;
            }
            
            // // Apply minimum zoom level
            // if (range < options.minTimeRange) {
            //     max = min + options.minTimeRange;
            //     needsUpdate = true;
            // }
            
            if (needsUpdate) {
                newLayout['xaxis.range'] = [min, max];
                Plotly.relayout(chartId, newLayout);
            }
        }
        
        // // Handle autorange resets
        // if (eventData['xaxis.autorange'] === true) {
        //     newLayout['xaxis.range'] = options.initialTimeRange;
        //     Plotly.relayout(chartId, newLayout);
        // }
    });
}

