// Constants
// const API_BASE_URL = 'http://localhost:25500/api/v1';
const API_BASE_URL = '/api/v1/upload';
const CHUNK_SIZE = 4 * 1024 * 1024; // 2MB chunks

// DOM Elements
const uploadContainer = document.getElementById('upload-container');
const fileInput = document.getElementById('audio-file');
const fileInfo = document.getElementById('file-info');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar-fill');
const progressText = document.getElementById('progress-text');
const resultContainer = document.getElementById('result-container');
const errorMessage = document.getElementById('error-message');
const successMessage = document.getElementById('success-message');
const processingSpinner = document.getElementById('processing-spinner');
const playPauseBtn = document.getElementById('playPauseBtn');
const volumeSlider = document.getElementById('volumeSlider');
const currentSpeaker = document.getElementById('current-speaker');
const currentTime = document.getElementById('current-time');
const diarizationMethod = document.getElementById('diarization_method');
const isLimitTime = document.getElementById('is_limit_time');
const limitTimeSec = document.getElementById('limit_time_sec');
const timeLimitValue = document.getElementById('time-limit-value');

// Audio and Waveform variables
let audioContext;
let audioBuffer;
let currentAudioSource;
let wavesurfer;
let currentSegmentIndex = -1;
let segments = [];
let isPlaying = false;
let uploadedFile = null;  // Store the uploaded file
let uploadStartTime = null;
let elapsedTimerInterval = null; // Added for elapsed time timer

// Add these variables at the top with other global variables
let isUserScrolling = false;
let scrollTimeout = null;
let currentAudioUuid = null;

// Add global variables to store intermediate results
let lastCompletedStep = "";

// Sentiment Analysis Functions
let sentimentBtn;
let analysisContainer;
let interactiveAnalysisBtn;
let charts = {};

// Initialize WaveSurfer instances
function initializeWaveSurfers() {
    // Initialize current segment waveform
    wavesurfer = WaveSurfer.create({
        container: '#waveform-current',
        waveColor: '#4a9eff',
        progressColor: '#2196F3',
        cursorColor: '#333',
        height: 80,
        normalize: true,
        backend: 'WebAudio',
        plugins: [],
        responsive: true,
        cursorWidth: 2,
        barWidth: 3,
        barGap: 1,
        interact: true,
        // Add duration limit if enabled
        maxLength: isLimitTime.checked ? parseInt(limitTimeSec.value) : undefined
    });

    // Event listeners for waveform
    wavesurfer.on('audioprocess', updateCurrentTime);
    wavesurfer.on('finish', () => {
        playPauseBtn.textContent = 'Play';
        isPlaying = false;
        playNextSegment();
    });
    
    wavesurfer.on('ready', () => {
        // Set initial volume
        const volume = volumeSlider.value / 100;
        wavesurfer.setVolume(volume);
    });

    // Add click event listener for waveform
    wavesurfer.on('click', (e) => {
        if (!segments.length) return;
        
        const clickTime = wavesurfer.getCurrentTime();
        const foundIndex = segments.findIndex(segment => 
            clickTime >= segment.start && clickTime <= segment.end
        );
        
        if (foundIndex !== -1) {
            currentSegmentIndex = foundIndex;
            highlightCurrentSegment();
            updateSpeakerInfo(segments[foundIndex]);
        }
    });

    // Add seeking event listener
    wavesurfer.on('seeking', (e) => {
        if (!segments.length) return;
        
        const seekTime = wavesurfer.getCurrentTime();
        const foundIndex = segments.findIndex(segment => 
            seekTime >= segment.start && seekTime <= segment.end
        );
        
        if (foundIndex !== -1) {
            currentSegmentIndex = foundIndex;
            highlightCurrentSegment();
            updateSpeakerInfo(segments[foundIndex]);
        }
    });

    // Add these event listeners
    wavesurfer.on('play', () => {
        playPauseBtn.textContent = 'Pause';
        isPlaying = true;
    });

    wavesurfer.on('pause', () => {
        playPauseBtn.textContent = 'Play';
        isPlaying = false;
    });
}

// Initialize audio context
function initializeAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeWaveSurfers();
    
    // File upload event listeners
    uploadContainer.addEventListener('dragover', handleDragOver);
    uploadContainer.addEventListener('dragleave', handleDragLeave);
    uploadContainer.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Audio control event listeners
    playPauseBtn.addEventListener('click', togglePlayPause);
    volumeSlider.addEventListener('input', handleVolumeChange);

    // Options event listeners
    limitTimeSec.addEventListener('input', () => {
        timeLimitValue.textContent = `${limitTimeSec.value}s`;
        updateWaveformDuration();
    });

    isLimitTime.addEventListener('change', () => {
        document.getElementById('time-limit-group').style.opacity = isLimitTime.checked ? '1' : '0.5';
        limitTimeSec.disabled = !isLimitTime.checked;
        updateWaveformDuration();
    });

    // Add scroll event listener to result container
    document.getElementById('result-container').addEventListener('scroll', handleUserScroll);

    // Check for UUID in URL parameters
    const uuid = getUrlParameter('uuid');
    if (uuid) {
        loadPreviousResult(uuid);
    }

    // Add event listener for Redo Proc button
    document.getElementById('redoProcBtn').addEventListener('click', redoProcessing);
    
    // Add event listener for Clear button
    document.getElementById('clearBtn').addEventListener('click', clearEverything);

    // Add event listener for sentiment analysis button
    sentimentBtn = document.getElementById('sentimentBtn');
    if (sentimentBtn) {
        // sentimentBtn.addEventListener('click', runSentimentAnalysis);
        sentimentBtn.addEventListener('click', () => {
            runSentimentAnalysis('analysis-results');
            showInteractiveAnalysis('analysis-results');
        });
    }

    // Add event listener for sentiment analysis button 2
    const sentimentBtn2 = document.getElementById('sentimentBtn2');
    if (sentimentBtn2) {
        sentimentBtn2.addEventListener('click', () => showInteractiveAnalysis('analysis-results'));
    }
});

// Audio Control Functions
function togglePlayPause() {
    if (!wavesurfer) return;
    
    if (isPlaying) {
        wavesurfer.pause();
        playPauseBtn.textContent = 'Play';
    } else {
        wavesurfer.play();
        playPauseBtn.textContent = 'Pause';
    }
    isPlaying = !isPlaying;
}

function handleVolumeChange() {
    const volume = volumeSlider.value / 100;
    if (wavesurfer) wavesurfer.setVolume(volume);
    document.querySelector('.volume-value').textContent = `${volumeSlider.value}%`;
}

function updateCurrentTime() {
    if (!wavesurfer) return;
    
    const currentTime = wavesurfer.getCurrentTime();
    const duration = wavesurfer.getDuration();
    const timeString = `${formatTime(currentTime)} / ${formatTime(duration)}`;
    document.getElementById('current-time').textContent = timeString;
    
    // Update current segment based on time
    updateCurrentSegment(currentTime);
}

function updateCurrentSegment(currentTime) {
    if (!segments.length) return;
    
    const newSegmentIndex = segments.findIndex(segment => 
        currentTime >= segment.start && currentTime <= segment.end
    );
    
    if (newSegmentIndex !== -1) {
        currentSegmentIndex = newSegmentIndex;
        highlightCurrentSegment();
        updateSpeakerInfo(segments[currentSegmentIndex]);
    }
}

function highlightCurrentSegment() {
    // Remove active class from all segments
    document.querySelectorAll('.segment').forEach(seg => seg.classList.remove('active'));
    
    // Add active class to current segment
    const currentSegment = document.querySelector(`[data-segment-index="${currentSegmentIndex}"]`);
    if (currentSegment) {
        currentSegment.classList.add('active');
        scrollToSegment(currentSegment);
    }
}

function scrollToSegment(segmentElement) {
    if (!segmentElement || isUserScrolling) return;
    
    const container = document.getElementById('result-container');
    const containerRect = container.getBoundingClientRect();
    const segmentRect = segmentElement.getBoundingClientRect();
    
    if (segmentRect.top < containerRect.top || segmentRect.bottom > containerRect.bottom) {
        container.scrollTop = segmentElement.offsetTop - container.offsetTop - (container.clientHeight / 2);
    }
}

function updateSpeakerInfo(segment) {
    let speakerRole;
    
    if (segment.speaker === 0) {
        speakerRole = 'Counselor';
    } else if (segment.speaker === 'undecided') {
        speakerRole = 'Undecided';
    } else {
        speakerRole = `Client ${segment.speaker}`;
    }
    
    currentSpeaker.textContent = `Speaker: ${speakerRole}`;
}

function playNextSegment() {
    if (currentSegmentIndex < segments.length - 1) {
        currentSegmentIndex++;
        const nextSegment = segments[currentSegmentIndex];
        wavesurfer.seekTo(nextSegment.start / wavesurfer.getDuration());
        wavesurfer.play();
        isPlaying = true;
        playPauseBtn.textContent = 'Pause';
    }
}

// Transcript Display Functions
async function displayResult(audioUuid) {
    try {
        // Now just calls displayStepResult with 'improving' step
        await displayStepResult(audioUuid, 'improving');
    } catch (error) {
        console.error('Error displaying result:', error);
        hideSpinner();
        showError('Error loading audio file: ' + error.message);
    }
}

// Playback Control Functions
function playSegment(index) {
    if (!wavesurfer || !segments[index]) return;
    
    currentSegmentIndex = index;
    const segment = segments[index];
    
    wavesurfer.seekTo(segment.start / wavesurfer.getDuration());
    wavesurfer.play();
    isPlaying = true;
    playPauseBtn.textContent = 'Pause';
    
    highlightCurrentSegment();
    updateSpeakerInfo(segment);
}

// File Upload Functions
function handleDragOver(e) {
    e.preventDefault();
    uploadContainer.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadContainer.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadContainer.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('audio/')) {
        showError('Please select an audio file');
        return;
    }
    
    uploadedFile = file;  // Store the file
    fileInfo.textContent = `Selected file: ${file.name} (${formatFileSize(file.size)})`;
    uploadChunks(file);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showProgress() {
    progressContainer.style.display = 'block';
    document.getElementById('steps-progress').style.display = 'none';
    updateProgress(0);
}

function updateProgress(percent) {
    progressBar.style.width = `${percent}%`;
    progressText.textContent = `${Math.round(percent)}%`;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    successMessage.style.display = 'none';
}

function showSuccess(message) {
    successMessage.textContent = message;
    successMessage.style.display = 'block';
    errorMessage.style.display = 'none';
}

function hideMessages() {
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
}

function showSpinner() {
    processingSpinner.style.display = 'block';
    document.getElementById('steps-progress').style.display = 'block';
}

function hideSpinner() {
    processingSpinner.style.display = 'none';
}

// Add this function to format elapsed time
function formatElapsedTime(seconds) {
    const totalSeconds = Math.floor(seconds);
    const minutes = Math.floor(totalSeconds / 60);
    const remainingSeconds = totalSeconds % 60;
    return `${minutes}min ${remainingSeconds}sec (${totalSeconds} sec)`;
}

// Add this function to update the elapsed time display
function updateElapsedTime() {
    if (!uploadStartTime) return;
    const now = Date.now();
    const elapsedSeconds = (now - uploadStartTime) / 1000;
    const elapsedTimeElement = document.getElementById('elapsed-time');
    if (elapsedTimeElement) {
        elapsedTimeElement.textContent = `Elapsed Time: ${formatElapsedTime(elapsedSeconds)}`;
    }
}

// Add this function to start the elapsed time timer
function startElapsedTimer() {
    // Clear any existing timer
    stopElapsedTimer(); 
    // Set start time if not already set (important for reprocess)
    if (!uploadStartTime) {
        uploadStartTime = Date.now();
    }
    // Update immediately and then every second
    updateElapsedTime(); 
    elapsedTimerInterval = setInterval(updateElapsedTime, 1000);
}

// Add this function to stop the elapsed time timer
function stopElapsedTimer() {
    if (elapsedTimerInterval) {
        clearInterval(elapsedTimerInterval);
        elapsedTimerInterval = null;
        // Optionally, update one last time
        updateElapsedTime(); 
    }
}

async function uploadChunks(file) {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const audioUuid = generateUUID();
    let uploadedChunks = 0;
    
    // Store current options
    const currentOptions = {
        diarization_method: diarizationMethod.value,
        is_limit_time: isLimitTime.checked,
        limit_time_sec: parseInt(limitTimeSec.value)
    };
    
    // Set start time when upload begins
    uploadStartTime = Date.now();
    
    showProgress();
    hideMessages();
    
    // Clear and ensure steps container is visible
    const stepsContainer = document.getElementById('steps-progress');
    stepsContainer.innerHTML = ''; // Clear previous steps
    stepsContainer.style.display = 'block'; // Force display
    
    try {
        const uploadPromises = [];
        
        for (let i = 0; i < totalChunks; i++) {
            const start = i * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, file.size);
            const chunk = file.slice(start, end);
            
            uploadPromises.push(
                uploadChunk(chunk, file.name, audioUuid, i, totalChunks)
                .then(() => {
                    uploadedChunks++;
                    updateProgress(uploadedChunks / totalChunks * 100);
                })
            );
        }
        
        await Promise.all(uploadPromises);

        // Update URL with UUID after successful upload
        updateUrlWithUuid(audioUuid);
        
        // Start timer *after* upload completes
        startElapsedTimer(); 

        showSpinner();
        progressContainer.style.display = 'none';

        // Show audio player container and load waveform immediately
        document.getElementById('audio-player-container').style.display = 'block';
        const audioUrl = URL.createObjectURL(file);
        await wavesurfer.load(audioUrl);

        await pollStatus(audioUuid);
        
        // Show the Redo Proc button
        document.getElementById('redoProcBtn').style.display = 'inline';
        
    } catch (error) {
        hideSpinner();
        showError('Upload failed: ' + error.message + ' ... ' +error);
    }
}

async function uploadChunk(chunk, originalFilename, audioUuid, chunkIndex, totalChunks) {
    const formData = new FormData();
    formData.append('file', chunk);
    formData.append('audio_uuid', audioUuid);
    formData.append('chunk_index', chunkIndex);
    formData.append('total_chunks', totalChunks);
    formData.append('original_filename', originalFilename);
    
    // Add options
    const options = {
        diarization_method: diarizationMethod.value,
        is_limit_time: isLimitTime.checked,
        limit_time_sec: parseInt(limitTimeSec.value)
    };
    formData.append('options', JSON.stringify(options));
    
    const response = await fetch(`${API_BASE_URL}/chunk`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

function displaySteps(steps) {
    console.log("Displaying steps:", steps);
    
    const stepsContainer = document.getElementById('steps-progress');
    stepsContainer.innerHTML = steps.map(step => {
        console.log("Processing step:", step);
        
        let progressHtml = '';
        let detailsHtml = '';
        
        // Fix any invalid step_name values
        // If step_name is one of the status values, replace it with a more specific value
        if (step.step_name === "in_progress" || step.step_name === "completed" || step.step_name === "failed") {
            console.warn(`Invalid step_name detected: "${step.step_name}"`);
            // Don't show steps with invalid names at all
            return ''; // Skip this step instead of showing a generic "processing" step
        }
        
        // Add progress bar for uploading step
        if (step.step_name === 'uploading') {
            const percent = (step.processed_chunks / step.total_chunks) * 100;
            progressHtml = `
                <div class="step-progress">
                    <div class="step-progress-fill" style="width: ${percent}%"></div>
                </div>
                <div class="step-details">
                    Processed ${step.processed_chunks} of ${step.total_chunks} chunks
                </div>
            `;
        } 
        // Add VAD visualization for splitting step
        else if (step.step_name === 'splitting' && step.vad_segments) {
            // Define colors for VAD segments
            const vadColors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63"];  // More colors
            
            // Add VAD visualization
            const vadHtml = `
                <div class="vad-visualization">
                    <div class="vad-timeline">
                        ${step.vad_segments.map((segment, idx) => {
                            const startPercent = (segment.start / step.total_duration) * 100;
                            const widthPercent = (segment.duration / step.total_duration) * 100;
                            const color = vadColors[idx % vadColors.length];
                            return `<div class="vad-segment" 
                                       style="left: ${startPercent}%; width: ${widthPercent}%; background-color: ${color}; opacity: 0.8; cursor: pointer;" 
                                       title="Segment ${idx + 1}: ${segment.duration.toFixed(2)}s"
                                       onclick="handleTimelineClick(${segment.start}, ${step.total_duration})"></div>`;
                        }).join('')}
                    </div>
                    <div class="vad-timeline-labels">
                        <span>0:00</span>
                        <span>${formatTime(step.total_duration)}</span>
                    </div>
                    <div class="vad-info">
                        <span>Found ${step.num_segments} sound segments</span>
                        <span>Total duration: ${formatTime(step.total_duration)}</span>
                    </div>
                </div>
            `;
            detailsHtml = vadHtml;
        } 
        // Add transcription progress for transcribing step
        else if (step.step_name === 'transcribing' && step.total_splits) {
            const percent = (step.processed_splits / step.total_splits) * 100;
            progressHtml = `
                <div class="step-progress">
                    <div class="step-progress-fill" style="width: ${percent}%"></div>
                </div>
                <div class="step-details">
                    <div class="transcription-progress">
                        <div>Transcribing splits: ${step.processed_splits} of ${step.total_splits}</div>
                        <div class="progress-percent">${percent.toFixed(1)}% complete</div>
                    </div>
                </div>
            `;
        } 
        // Add diarization visualization for diarizing step
        else if (step.step_name === 'diarizing' && step.diarization_segments) {
            // Define speaker colors (up to 5 speakers)
            const speakerColors = [
                "#2196F3 ",  // Blue - A
                "#4CAF50",  // Green - B
                "#FF9800",  // Orange - C
                "#9C27B0",  // Purple - D
                "#E91E63"   // Pink - E
            ];
            
            // Number of speakers (limit to 5 max for display)
            const numSpeakers = Math.min(step.num_speakers || 0, 5);
            
            // Add diarization visualization
            const diarHtml = `
                <div class="vad-visualization">
                    <div class="vad-timeline">
                        ${step.diarization_segments.map(segment => {
                            const startPercent = (segment.start / step.total_duration) * 100;
                            const widthPercent = (segment.duration / step.total_duration) * 100;
                            const speakerIdx = parseInt(segment.speaker);
                            const color = speakerColors[speakerIdx] || "#9E9E9E";
                            const speakerLetter = speakerIdx < 5 ? String.fromCharCode(65 + speakerIdx) : '?';
                            return `<div class="vad-segment" 
                                       style="left: ${startPercent}%; width: ${widthPercent}%; background-color: ${color} !important; cursor: pointer;" 
                                       title="Speaker ${speakerLetter}: ${segment.duration.toFixed(2)}s"
                                       onclick="handleTimelineClick(${segment.start}, ${step.total_duration})"></div>`;
                        }).join('')}
                    </div>
                    <div class="vad-timeline-labels">
                        <span>0:00</span>
                        <span>${formatTime(step.total_duration)}</span>
                    </div>
                    <div class="vad-info">
                        <div class="speaker-legend">
                            ${Array.from({length: numSpeakers}, (_, i) => {
                                const letter = String.fromCharCode(65 + i);
                                return `
                                    <span class="speaker-item">
                                        <span class="speaker-color" style="background-color: ${speakerColors[i]}"></span>
                                        ${letter}
                                    </span>
                                `;
                            }).join('')}
                        </div>
                        <span>Found ${step.num_speakers} speakers</span>
                    </div>
                </div>
            `;
            detailsHtml = diarHtml;
        } 
        // For steps without special visualization that aren't failed
        else if (step.status !== 'failed') {
            // Add empty details div
            detailsHtml = `<div class="step-details"></div>`;
        }

        // Add error details if failed
        if (step.status === 'failed' && step.error) {
            detailsHtml = `<div class="step-details error-text">${step.error}</div>`;
        }

        // Add CSS class based on status ("in_progress", "completed", or "failed")
        // But also add step_name as a secondary class for specific styling
        return `
            <div class="step ${step.status} ${step.step_name}">
                <div class="step-header">
                    <span class="step-name">${formatStepName(step.step_name)}</span>
                    <span class="step-timestamp">${formatTimestamp(step.timestamp)}</span>
                </div>
                ${progressHtml}
                ${detailsHtml}
                <div class="step-status-indicator">${formatStepStatus(step.status)}</div>
            </div>
        `;
    }).filter(html => html !== '').join(''); // Filter out empty strings
}

// Add this function to format step_name similarly to status
function formatStepName(stepName) {
    if (!stepName) return 'Unknown';
    return stepName.charAt(0).toUpperCase() + stepName.slice(1).toLowerCase();
}

function formatStepStatus(status) {
    if (!status) return 'Unknown';
    
    // Ensure we display "in_progress" as "In Progress" (with space) for display
    if (status === "in_progress") return "In Progress";
    
    return status.charAt(0).toUpperCase() + status.slice(1).toLowerCase();
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

async function pollStatus(audioUuid) {
    // Track which results we've already fetched
    const fetchedResults = {
        transcription: false,
        improving: false
    };
    
    // Add variables for tracking retries with exponential backoff
    let retryCount = 0;
    const maxRetries = 100;
    let retryDelay = 2000;
    
    while (true) {
        try {
            // Use fetch with timeout to avoid hanging
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
            
            const response = await fetch(`${API_BASE_URL}/status/${audioUuid}`, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId); // Clear timeout if fetch completed
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Reset retry count and delay on successful response
            retryCount = 0;
            retryDelay = 500;
            
            // Make sure the steps container is visible
            document.getElementById('steps-progress').style.display = 'block';
            
            // Display steps progress
            if (data.progress && data.progress.steps) {
                displaySteps(data.progress.steps);
                
                // Check for completed steps intelligently
                const completedSteps = findCompletedSteps(data.progress.steps);
                console.log("Completed steps:", completedSteps);
                
                // Only fetch transcription result once
                if (completedSteps.includes('transcribing') && !fetchedResults.transcription) {
                    console.log("Transcription completed, fetching results");
                    try {
                        await displayStepResult(audioUuid, 'transcription');
                        fetchedResults.transcription = true;
                        showSuccess("Initial transcription available - Speaker identification in progress");
                    } catch (err) {
                        console.log("Could not fetch transcription yet, will retry:", err);
                    }
                }
                
                // Only fetch improving result once - ONLY when improving step is completed, not when status is completing
                if (completedSteps.includes('improving') && !fetchedResults.improving) {
                    console.log("Improving completed, fetching final results");
                    try {
                        await displayStepResult(audioUuid, 'improving');
                        fetchedResults.improving = true;
                        showSuccess("Final improved transcription with speaker identification is now available.");
                    } catch (err) {
                        console.log("Could not fetch improving result yet, will retry:", err);
                    }
                }
            }
            
            // If status is completing, just finalize the process - don't re-fetch results
            if (data.status === 'completing') {
                // Try to fetch improving results if we haven't yet
                if (!fetchedResults.improving) {
                    try {
                        await displayStepResult(audioUuid, 'improving');
                        fetchedResults.improving = true;
                        hideSpinner();
                        stopElapsedTimer(); // Stop timer on completion
                        showSuccess(`Processing completed successfully (Total time: ${formatElapsedTime((Date.now() - uploadStartTime) / 1000)})`);
                        break;
                    } catch (err) {
                        console.log("Could not fetch improving result yet:", err);
                        // If we've been trying for too long, give up and use transcription
                        if (retryCount >= maxRetries) {
                            hideSpinner();
                            stopElapsedTimer(); // Stop timer on give up
                            if (fetchedResults.transcription) {
                                showSuccess("Processing completed with initial transcription only.");
                            } else {
                                showError("Could not fetch final results.");
                            }
                            break;
                        }
                    }
                } else {
                    hideSpinner();
                    stopElapsedTimer(); // Stop timer on completion
                    showSuccess(`Processing completed successfully (Total time: ${formatElapsedTime((Date.now() - uploadStartTime) / 1000)})`);
                    break;
                }
            } else if (data.status === 'failed') {
                hideSpinner();
                stopElapsedTimer(); // Stop timer on failure
                throw new Error(data.error || 'Processing failed');
            }
            
            // Normal polling delay (1 second)
            await new Promise(resolve => setTimeout(resolve, 1000));
            
        } catch (error) {
            console.error(`Error checking status (attempt ${retryCount + 1}):`, error.message);
            
            // Implement exponential backoff for retries
            retryCount++;
            
            if (retryCount > maxRetries) {
                // After max retries, show error and stop polling
                hideSpinner();
                stopElapsedTimer(); // Stop timer on max retries failure
                showError(`Failed to get status after ${maxRetries} attempts: ${error.message}`);
                break;
            }
            
            // Exponential backoff: 500ms, 1s, 2s, 4s, 8s
            retryDelay = Math.min(retryDelay * 2, 8000);
            console.log(`Retrying in ${retryDelay}ms...`);
            await new Promise(resolve => setTimeout(resolve, retryDelay));
        }
    }
}

// Helper function to find completed steps
function findCompletedSteps(steps) {
    return steps
        .filter(step => step.status === 'completed')
        .map(step => step.step_name);
}

// Add this function to display a speaker legend
function updateSpeakerLegend() {
    // Get all unique speakers from segments
    const speakers = new Set();
    segments.forEach(segment => {
        if (segment.speaker !== 'undecided') {
            speakers.add(segment.speaker);
        }
    });
    
    if (speakers.size === 0) return;
    
    // Create the legend HTML
    let legendHtml = '<div class="speaker-legend-container">';
    legendHtml += '<h3>Speakers</h3>';
    legendHtml += '<div class="speaker-legend">';
    
    // Add a legend item for each speaker
    speakers.forEach(speaker => {
        let speakerClass;
        let speakerLabel;
        
        if (speaker === 0) {
            speakerClass = 'counselor';
            speakerLabel = 'Counselor';
        } else {
            speakerClass = `speaker-${speaker}`;
            speakerLabel = `Client ${speaker}`;
        }
        
        legendHtml += `
            <div class="speaker-legend-item">
                <div class="speaker-color-sample ${speakerClass}"></div>
                <div class="speaker-name">${speakerLabel}</div>
            </div>
        `;
    });
    
    legendHtml += '</div></div>';
    
    // Add the legend to the result header
    const resultHeader = document.querySelector('.result-header');
    
    // Check if legend already exists
    const existingLegend = document.querySelector('.speaker-legend-container');
    if (existingLegend) {
        existingLegend.remove();
    }
    
    // Add the new legend after the header content
    resultHeader.insertAdjacentHTML('beforeend', legendHtml);
}

// Enhanced transcript display function with better error handling
function updateTranscriptDisplay() {
    if (!segments || segments.length === 0) {
        console.warn("No segments to display");
        document.getElementById('transcript').innerHTML = '<div class="error-text">No transcript segments available</div>';
        return;
    }

    console.log("Updating transcript display with segments:", segments);

    num_speakers = segments.map(segment => segment.speaker).filter(speaker => speaker !== 'undecided').length;
    console.log("Number of speakers:", num_speakers);
    
    const transcriptHtml = segments.map((segment, index) => {
        let speakerClass = 'undecided';
        let speakerLabel = 'Undecided';
        
        if (segment.speaker !== 'undecided') {
            // Handle numeric speaker IDs for different styling
            if (segment.speaker === 0) {
                speakerClass = 'counselor';
                speakerLabel = 'Counselor';
            } else {
                if(num_speakers == 2){
                    speakerClass = `speaker-${segment.speaker}`;
                    speakerLabel = `Client`;
                }else{
                    // All other speakers are clients with different colors
                    speakerClass = `speaker-${segment.speaker}`;
                    speakerLabel = `Client ${segment.speaker}`;
                }
            }
        }
        
        return `
            <div class="segment ${speakerClass}" 
                 data-segment-index="${index}"
                 onclick="playSegment(${index})">
                <div class="segment-speaker">${speakerLabel}</div>
                <div class="segment-time">${formatTime(segment.start)} - ${formatTime(segment.end)}</div>
                <div class="segment-text">${segment.text}</div>
            </div>
        `;
    }).join('');
    
    document.getElementById('transcript').innerHTML = transcriptHtml;
    
    // Update the speaker legend
    updateSpeakerLegend();
}

// Modified displayStepResult function to update legend
async function displayStepResult(audioUuid, stepName) {
    try {
        console.log(`Fetching ${stepName} result for ${audioUuid}`);
        const response = await fetch(`${API_BASE_URL}/result/${stepName}/${audioUuid}`);
        if (!response.ok) {
            console.error(`Error fetching ${stepName} result: ${response.status} ${response.statusText}`);
            return;
        }
        
        const data = await response.json();
        console.log(`Got ${stepName} result:`, data);
        
        if (data && data.result) {
            // Store segments based on step
            segments = processStepResult(data.result, stepName);
            console.log(`Processed ${segments.length} segments for ${stepName}`);
            
            // Make sure we have segments with text
            if (!segments || segments.length === 0) {
                console.error(`No valid segments found in ${stepName} result`);
                return;
            }
            
            // Display transcript immediately
            updateTranscriptDisplay();
            
            // Show result container
            resultContainer.style.display = 'block';
            
            // Add notice about result type
            let resultNotice = '';
            if (stepName === 'transcription') {
                resultNotice = '<div class="result-notice">Showing initial transcription. Speaker identification in progress...</div>';
            } else if (stepName === 'improving') {
                resultNotice = '<div class="result-notice">Improved transcription with speaker identification complete.</div>';
            }
            
            // Find the header element
            const resultHeader = document.querySelector('.result-header');
            // Remove existing notice if it exists
            const existingNotice = resultHeader.querySelector('.result-notice');
            if (existingNotice) {
                existingNotice.remove();
            }
            // Append the new notice
            if (resultNotice) {
                resultHeader.insertAdjacentHTML('beforeend', resultNotice);
            }
            
            // Show audio player container if not already visible
            document.getElementById('audio-player-container').style.display = 'block';
            
            // Update UI to reflect current step
            if (stepName === 'transcription') {
                document.querySelector('.result-container').classList.add('transcription-only');
                document.querySelector('.result-container').classList.remove('complete-result');
            } else if (stepName === 'improving') {
                document.querySelector('.result-container').classList.remove('transcription-only');
                document.querySelector('.result-container').classList.add('complete-result');
            }
        }
    } catch (error) {
        console.error(`Error displaying ${stepName} result:`, error);
    }
}

// Enhanced function to better process the step results
function processStepResult(result, stepName) {
    console.log(`Processing ${stepName} result:`, result);
    
    if (!result || !result.segments || !Array.isArray(result.segments)) {
        console.error(`Invalid result format for ${stepName}:`, result);
        return [];
    }
    
    if (stepName === 'transcription') {
        // For transcription, all speakers are "undecided"
        return result.segments.map(segment => ({
            ...segment,
            speaker: segment.speaker || 'undecided'
        }));
    } else if (stepName === 'improving') {
        // For improving, use actual speaker IDs and ensure all fields are present
        return result.segments.map(segment => ({
            start: segment.start || 0,
            end: segment.end || 0,
            text: segment.text || '',
            speaker: segment.speaker !== undefined ? segment.speaker : 'undecided'
        }));
    }
    return [];
}

// Helper Functions
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Add cleanup function for when leaving the page
window.addEventListener('beforeunload', () => {
    if (uploadedFile) {
        URL.revokeObjectURL(uploadedFile);
    }
});

// Add this function to handle user scrolling
function handleUserScroll() {
    isUserScrolling = true;
    
    // Clear existing timeout
    if (scrollTimeout) {
        clearTimeout(scrollTimeout);
    }
    
    // Set new timeout
    scrollTimeout = setTimeout(() => {
        isUserScrolling = false;
    }, 1000); // Wait 1 second after scrolling stops
}

// Add this function to handle timeline segment clicks
function handleTimelineClick(startTime, totalDuration) {
    if (!wavesurfer) return;
    
    const normalizedTime = startTime / totalDuration;
    wavesurfer.seekTo(normalizedTime);
    wavesurfer.play();
}

// Add this function to create a trimmed audio blob
async function createTrimmedAudioBlob(audioFile, duration) {
    // Create audio context
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Read the file
    const arrayBuffer = await audioFile.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Calculate the number of samples for the desired duration
    const sampleRate = audioBuffer.sampleRate;
    const numberOfSamples = Math.min(Math.floor(duration * sampleRate), audioBuffer.length);
    
    // Create a new buffer for the trimmed audio
    const trimmedBuffer = audioContext.createBuffer(
        audioBuffer.numberOfChannels,
        numberOfSamples,
        sampleRate
    );
    
    // Copy the samples
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        const channelData = audioBuffer.getChannelData(channel);
        const trimmedData = trimmedBuffer.getChannelData(channel);
        trimmedData.set(channelData.slice(0, numberOfSamples));
    }
    
    // Convert buffer to wave file
    const wavBlob = await audioBufferToWave(trimmedBuffer);
    return new Blob([wavBlob], { type: 'audio/wav' });
}

// Add this helper function to convert AudioBuffer to Wave format
function audioBufferToWave(audioBuffer) {
    const numOfChan = audioBuffer.numberOfChannels;
    const length = audioBuffer.length * numOfChan * 2;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);
    const channels = [];
    let pos = 0;
    let offset = 0;

    // Write WAVE header
    setUint32(0x46464952);                         // "RIFF"
    setUint32(36 + length);                        // file length
    setUint32(0x45564157);                         // "WAVE"
    setUint32(0x20746d66);                         // "fmt " chunk
    setUint32(16);                                 // length = 16
    setUint16(1);                                  // PCM (uncompressed)
    setUint16(numOfChan);
    setUint32(audioBuffer.sampleRate);
    setUint32(audioBuffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
    setUint16(numOfChan * 2);                      // block-align
    setUint16(16);                                 // 16-bit
    setUint32(0x61746164);                         // "data" - chunk
    setUint32(length);                             // chunk length

    // Write interleaved data
    for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
        channels.push(audioBuffer.getChannelData(i));
    }

    while (pos < length) {
        for (let i = 0; i < numOfChan; i++) {
            let sample = Math.max(-1, Math.min(1, channels[i][offset]));
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
            view.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }

    function setUint16(data) {
        view.setUint16(pos, data, true);
        pos += 2;
    }

    function setUint32(data) {
        view.setUint32(pos, data, true);
        pos += 4;
    }

    return buffer;
}

// Modify the updateWaveformDuration function
async function updateWaveformDuration() {
    if (!wavesurfer || !uploadedFile) return;

    try {
        let audioUrl;
        if (isLimitTime.checked) {
            const timeLimit = parseInt(limitTimeSec.value);
            // Create a trimmed version of the audio
            const trimmedBlob = await createTrimmedAudioBlob(uploadedFile, timeLimit);
            audioUrl = URL.createObjectURL(trimmedBlob);
        } else {
            audioUrl = URL.createObjectURL(uploadedFile);
        }

        // Load the appropriate audio into wavesurfer
        await wavesurfer.load(audioUrl);

        // Clean up the URL after loading
        if (audioUrl !== URL.createObjectURL(uploadedFile)) {
            URL.revokeObjectURL(audioUrl);
        }
    } catch (error) {
        console.error('Error updating waveform duration:', error);
        showError('Error updating waveform visualization');
    }
}

// Add this function to get URL parameters
function getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

// Add this function to update URL with UUID
function updateUrlWithUuid(uuid) {
    const newUrl = `${window.location.pathname}?uuid=${uuid}`;
    window.history.pushState({ uuid }, '', newUrl);
    currentAudioUuid = uuid;
}

// Add this function to restore analysis options
function restoreAnalysisOptions(options) {
    if (!options) return;
    
    // Restore diarization method
    if (options.diarization_method) {
        const methodSelect = document.getElementById('diarization_method');
        methodSelect.value = options.diarization_method;
    }
    
    // Restore time limit settings
    if (options.is_limit_time !== undefined) {
        const isLimitTimeCheckbox = document.getElementById('is_limit_time');
        isLimitTimeCheckbox.checked = options.is_limit_time;
        
        // Update time limit group visibility
        document.getElementById('time-limit-group').style.opacity = options.is_limit_time ? '1' : '0.5';
        limitTimeSec.disabled = !options.is_limit_time;
    }
    
    // Restore time limit value
    if (options.limit_time_sec) {
        const limitTimeSecInput = document.getElementById('limit_time_sec');
        limitTimeSecInput.value = options.limit_time_sec;
        document.getElementById('time-limit-value').textContent = `${options.limit_time_sec}s`;
    }
}

// Modify loadPreviousResult to handle missing audio files
async function loadPreviousResult(uuid) {
    try {
        showSpinner();
        currentAudioUuid = uuid;
        uploadStartTime = null; // Reset start time for loaded results

        // First check the status
        const statusResponse = await fetch(`${API_BASE_URL}/status/${uuid}`);
        if (!statusResponse.ok) {
            throw new Error('Failed to fetch status');
        }
        const statusData = await statusResponse.json();

        // Try to get start time from the first step
        if (statusData.progress && statusData.progress.steps && statusData.progress.steps.length > 0) {
            uploadStartTime = new Date(statusData.progress.steps[0].timestamp).getTime();
        }
        
        // Restore analysis options
        if (statusData.options) {
            restoreAnalysisOptions(statusData.options);
        }

        // Display steps progress
        if (statusData.progress && statusData.progress.steps) {
            displaySteps(statusData.progress.steps);
        }

        // Show audio player container
        document.getElementById('audio-player-container').style.display = 'block';

        // Try to load audio file for playback
        try {
            const audioUrl = `${API_BASE_URL}/audio/${uuid}`;
            await wavesurfer.load(audioUrl);
        } catch (audioError) {
            console.error('Error loading audio file:', audioError);
            showWarning("Audio file not found, but transcript is available");
            // Continue with loading transcript even if audio file is missing
        }

        // If completed or completing, fetch the final result
        if (statusData.status === 'completing' || statusData.status === 'completed') {
            await displayStepResult(uuid, 'improving');
            hideSpinner();
            stopElapsedTimer(); // Stop timer if already completed
            updateElapsedTime(); // Show final time
            showSuccess("Previous analysis result loaded successfully");
        } else if (statusData.status === 'failed') {
            hideSpinner();
            stopElapsedTimer(); // Stop timer if failed
            updateElapsedTime(); // Show final time
            showError(statusData.error || 'Processing failed');
        } else {
            // If still processing, start polling
            startElapsedTimer(); // Start timer for ongoing process
            await pollStatus(uuid);
        }
        
        // Show the Redo Proc button
        document.getElementById('redoProcBtn').style.display = 'inline';
        
    } catch (error) {
        console.error('Error loading previous result:', error);
        hideSpinner();
        showError('Error loading previous result: ' + error.message);
    }
}

// Add a warning function
function showWarning(message) {
    const warningDiv = document.getElementById('warning-message') || 
        (() => {
            const div = document.createElement('div');
            div.id = 'warning-message';
            div.className = 'warning-message';
            document.querySelector('.upload-section').appendChild(div);
            return div;
        })();
    
    warningDiv.textContent = message;
    warningDiv.style.display = 'block';
}

// Update this function to handle redoing the processing without a spinner
async function redoProcessing() {
    if (!currentAudioUuid) {
        showError("No audio file available to reprocess");
        return;
    }

    try {
        // Set start time when reprocessing begins
        uploadStartTime = Date.now();
        startElapsedTimer(); // Start timer immediately
        
        // Clear messages
        hideMessages();
        
        // Hide spinner explicitly (in case it's visible)
        processingSpinner.style.display = 'none';
        
        // Clear previous results
        document.getElementById('transcript').innerHTML = '';
        
        // Clear and ensure steps container is visible
        const stepsContainer = document.getElementById('steps-progress');
        stepsContainer.innerHTML = ''; // Clear previous steps
        stepsContainer.style.display = 'block'; // Force display
        
        // Make API call to restart processing - NOT awaited
        fetch(`${API_BASE_URL}/reprocess/${currentAudioUuid}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                options: {
                    diarization_method: diarizationMethod.value,
                    is_limit_time: isLimitTime.checked,
                    limit_time_sec: parseInt(limitTimeSec.value)
                }
            })
        }).then(response => {
            if (!response.ok) {
                throw new Error(`Failed to restart processing: ${response.status} ${response.statusText}`);
            }
        }).catch(error => {
            console.error('Error reprocessing audio:', error);
            hideSpinner();
            showError('Error reprocessing audio: ' + error.message);
        });
        
        // Start polling for status updates immediately, without waiting for reprocess to complete
        setTimeout(() => { //fall one second later, after reprocess has been called
            pollStatus(currentAudioUuid);
        }, 1000);
        
    } catch (error) {
        console.error('Error initiating reprocessing:', error);
        hideSpinner();
        showError('Error initiating reprocessing: ' + error.message);
    }
}

// Add this function to clear all UI state
function clearEverything() {
    // Reset audio UUID
    currentAudioUuid = null;
    
    // Remove UUID from URL
    window.history.pushState({}, '', window.location.pathname);
    
    // Clear transcript
    document.getElementById('transcript').innerHTML = '';
    
    // Clear steps container
    const stepsContainer = document.getElementById('steps-progress');
    stepsContainer.innerHTML = '';
    stepsContainer.style.display = 'none';
    
    // Clear progress
    progressContainer.style.display = 'none';
    
    // Hide audio player container
    document.getElementById('audio-player-container').style.display = 'none';
    
    // Hide success and error messages
    hideMessages();
    
    // Hide spinner
    hideSpinner();
    
    // Hide Redo Proc button
    // document.getElementById('redoProcBtn').style.display = 'none';
    
    // Clear file info
    fileInfo.textContent = '';
    
    // Clear result container
    resultContainer.style.display = 'none';
    
    // Reset segments array
    segments = [];
    
    // Reset uploaded file
    uploadedFile = null;
    
    // Reset wavesurfer if it exists
    if (wavesurfer) {
        wavesurfer.empty();
    }
    
    // Reset file input value - IMPORTANT: This allows selecting the same file again
    fileInput.value = '';
    
    // Show success message
    showSuccess("All data has been cleared. You can now upload a new file.");
    
    // Stop timer on clear
    stopElapsedTimer(); 
    const elapsedTimeElement = document.getElementById('elapsed-time');
    if (elapsedTimeElement) {
        elapsedTimeElement.textContent = 'Elapsed Time: --';
    }
}
