class QSMusicApp {
    constructor() {
        this.config = {
            apiHost: 'http://127.0.0.1:8001',
            apiKey: '',
            pollInterval: 5000,
            requestTimeout: 300000
        };
        this.tasks = new Map();
        this.pollTimer = null;
        this.currentTaskId = null;
        this.refAudioFile = null;
        this.srcAudioFile = null;
        this.init();
    }

    init() {
        this.loadSettings();
        this.bindEvents();
        this.setupSliders();
        this.checkServerStatus();
        this.loadModels();
        this.startPolling();
        this.loadPresets();
    }

    loadSettings() {
        const saved = localStorage.getItem('qsmusic_settings');
        if (saved) {
            const settings = JSON.parse(saved);
            this.config = { ...this.config, ...settings };
            document.getElementById('apiHost').value = this.config.apiHost;
            document.getElementById('apiKey').value = this.config.apiKey;
            document.getElementById('pollInterval').value = this.config.pollInterval / 1000;
            document.getElementById('requestTimeout').value = this.config.requestTimeout / 1000;
        }
    }

    saveSettings() {
        this.config.apiHost = document.getElementById('apiHost').value;
        this.config.apiKey = document.getElementById('apiKey').value;
        this.config.pollInterval = parseInt(document.getElementById('pollInterval').value) * 1000;
        this.config.requestTimeout = parseInt(document.getElementById('requestTimeout').value) * 1000;
        localStorage.setItem('qsmusic_settings', JSON.stringify(this.config));
        this.showToast('设置已保存', 'success');
        this.checkServerStatus();
    }

    bindEvents() {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => this.switchTab(e));
        });

        document.querySelectorAll('.training-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTrainTab(e));
        });

        document.querySelectorAll('.collapsible').forEach(header => {
            header.addEventListener('click', (e) => this.toggleCollapse(e));
        });

        document.getElementById('refreshBtn').addEventListener('click', () => this.refresh());
        document.getElementById('quickGenerate').addEventListener('click', () => this.quickGenerate());
        document.getElementById('generateBtn').addEventListener('click', () => this.generateMusic());
        document.getElementById('randomSampleBtn').addEventListener('click', () => this.getRandomSample());
        document.getElementById('formatInputBtn').addEventListener('click', () => this.formatInput());
        document.getElementById('saveSettingsBtn').addEventListener('click', () => this.saveSettings());
        document.getElementById('testConnectionBtn').addEventListener('click', () => this.testConnection());
        document.getElementById('refreshModelsBtn').addEventListener('click', () => this.loadModels());
        document.getElementById('downloadModelBtn').addEventListener('click', () => this.downloadModel());
        document.getElementById('refreshLorasBtn').addEventListener('click', () => this.loadLoras());
        document.getElementById('refreshTasksBtn').addEventListener('click', () => this.refreshTasks());
        document.getElementById('clearTasksBtn').addEventListener('click', () => this.clearCompletedTasks());
        document.getElementById('startTrainBtn').addEventListener('click', () => this.startTraining('lora'));
        document.getElementById('startLokrTrainBtn').addEventListener('click', () => this.startTraining('lokr'));
        document.getElementById('estimateBtn').addEventListener('click', () => this.estimateResources());
        document.getElementById('stopTrainBtn').addEventListener('click', () => this.stopTraining());
        document.getElementById('importDataBtn').addEventListener('click', () => this.importData());
        document.getElementById('preprocessBtn').addEventListener('click', () => this.preprocessData());
        document.getElementById('genRandomSeed').addEventListener('change', (e) => this.toggleSeedInput(e));

        this.setupFileUpload('refAudioUpload', 'refAudioFile', 'refAudioInfo', 'refAudioName', 'refAudioFile');
        this.setupFileUpload('srcAudioUpload', 'srcAudioFile', 'srcAudioInfo', 'srcAudioName', 'srcAudioFile');

        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.applyPreset(e.target.dataset.preset));
        });

        document.getElementById('trainPreset').addEventListener('change', (e) => this.applyTrainPreset(e.target.value));
    }

    setupSliders() {
        const sliders = [
            { id: 'genDuration', display: 'durationValue' },
            { id: 'genBatch', display: 'batchValue' },
            { id: 'genSteps', display: 'stepsValue' },
            { id: 'genGuidance', display: 'guidanceValue' },
            { id: 'genShift', display: 'shiftValue' },
            { id: 'genLmTemp', display: 'lmTempValue' },
            { id: 'genLmCfg', display: 'lmCfgValue' },
            { id: 'genLmTopP', display: 'lmTopPValue' },
            { id: 'genLmRepPen', display: 'lmRepPenValue' },
            { id: 'genCoverStrength', display: 'coverStrengthValue' },
            { id: 'genCfgStart', display: 'cfgStartValue' },
            { id: 'genCfgEnd', display: 'cfgEndValue' },
            { id: 'loraRank', display: 'loraRankValue' },
            { id: 'loraAlpha', display: 'loraAlphaValue' },
            { id: 'loraDropout', display: 'loraDropoutValue' },
            { id: 'trainGradAcc', display: 'gradAccValue' },
            { id: 'preprocessSliceDur', display: 'sliceDurValue' },
            { id: 'preprocessSliceOverlap', display: 'sliceOverlapValue' }
        ];

        sliders.forEach(({ id, display }) => {
            const slider = document.getElementById(id);
            const displayEl = document.getElementById(display);
            if (slider && displayEl) {
                slider.addEventListener('input', (e) => {
                    displayEl.textContent = e.target.value;
                });
            }
        });
    }

    setupFileUpload(areaId, inputId, infoId, nameId, fileKey) {
        const area = document.getElementById(areaId);
        const input = document.getElementById(inputId);
        const info = document.getElementById(infoId);
        const name = document.getElementById(nameId);

        if (!area || !input) return;

        area.addEventListener('click', () => input.click());
        
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        });

        area.addEventListener('dragleave', () => {
            area.classList.remove('dragover');
        });

        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect(e.dataTransfer.files[0], info, name, fileKey);
            }
        });

        input.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0], info, name, fileKey);
            }
        });
    }

    handleFileSelect(file, infoEl, nameEl, fileKey) {
        if (fileKey === 'refAudioFile') {
            this.refAudioFile = file;
        } else if (fileKey === 'srcAudioFile') {
            this.srcAudioFile = file;
        }
        nameEl.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        infoEl.style.display = 'block';
    }

    switchTab(e) {
        const tab = e.currentTarget.dataset.tab;
        
        document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
        e.currentTarget.classList.add('active');

        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(`tab-${tab}`).classList.add('active');

        const titles = {
            generate: { title: '音乐生成', desc: '使用AI生成高质量音乐作品' },
            training: { title: '模型训练', desc: '训练自定义LoRA模型学习音乐风格' },
            dataset: { title: '数据集管理', desc: '管理训练数据集和预处理' },
            models: { title: '模型管理', desc: '管理和下载AI模型' },
            tasks: { title: '任务监控', desc: '查看生成和训练任务状态' },
            settings: { title: '系统设置', desc: '配置服务器和系统参数' }
        };

        document.getElementById('pageTitle').textContent = titles[tab].title;
        document.getElementById('pageDesc').textContent = titles[tab].desc;

        if (tab === 'models') {
            this.loadLoras();
        }
        if (tab === 'tasks') {
            this.refreshTasks();
        }
    }

    switchTrainTab(e) {
        const tab = e.currentTarget.dataset.trainTab;
        
        document.querySelectorAll('.training-tab').forEach(item => item.classList.remove('active'));
        e.currentTarget.classList.add('active');

        document.querySelectorAll('.train-panel').forEach(panel => panel.style.display = 'none');
        document.getElementById(`train-${tab}`).style.display = 'block';
    }

    toggleCollapse(e) {
        const header = e.currentTarget;
        const targetId = header.dataset.collapse;
        const body = document.getElementById(`collapse-${targetId}`);
        const icon = header.querySelector('.collapse-icon');

        if (body.style.display === 'none') {
            body.style.display = 'block';
            icon.classList.remove('collapsed');
        } else {
            body.style.display = 'none';
            icon.classList.add('collapsed');
        }
    }

    toggleSeedInput(e) {
        const seedRow = document.getElementById('seedRow');
        seedRow.style.display = e.target.checked ? 'none' : 'flex';
    }

    async apiRequest(endpoint, options = {}) {
        const url = `${this.config.apiHost}${endpoint}`;
        const headers = {
            ...options.headers
        };

        if (this.config.apiKey) {
            headers['Authorization'] = `Bearer ${this.config.apiKey}`;
        }

        if (!(options.body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
        }

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.config.requestTimeout);

            const response = await fetch(url, {
                ...options,
                headers,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const error = await response.text();
                throw new Error(error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            if (data.code !== 200) {
                throw new Error(data.error || 'API请求失败');
            }
            return data.data;
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('请求超时');
            }
            throw error;
        }
    }

    async checkServerStatus() {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.server-status span');
        
        try {
            await this.apiRequest('/health');
            statusDot.className = 'status-dot connected';
            statusText.textContent = '服务器已连接';
            return true;
        } catch (error) {
            statusDot.className = 'status-dot disconnected';
            statusText.textContent = '服务器未连接';
            return false;
        }
    }

    async testConnection() {
        this.saveSettings();
        const connected = await this.checkServerStatus();
        if (connected) {
            this.showToast('连接成功！', 'success');
            this.loadModels();
        } else {
            this.showToast('连接失败，请检查服务器地址', 'error');
        }
    }

    async loadModels() {
        try {
            const data = await this.apiRequest('/v1/models');
            const select = document.getElementById('genModel');
            const grid = document.getElementById('modelGrid');

            select.innerHTML = '<option value="">默认模型</option>';
            grid.innerHTML = '';

            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name + (model.is_default ? ' (默认)' : '');
                if (model.is_default) option.selected = true;
                select.appendChild(option);

                const card = this.createModelCard(model);
                grid.appendChild(card);
            });
        } catch (error) {
            console.error('加载模型失败:', error);
        }
    }

    createModelCard(model) {
        const card = document.createElement('div');
        card.className = 'model-card';
        card.innerHTML = `
            <h4>🤖 ${model.name}</h4>
            <p>DiT扩散模型，用于音乐生成</p>
            <div class="model-meta">
                <span>类型: DiT</span>
            </div>
            <span class="model-status installed">✓ 已加载</span>
            <div class="model-actions" style="margin-top: 12px;">
                <button class="btn btn-secondary" onclick="app.selectModel('${model.name}')">选择使用</button>
            </div>
        `;
        return card;
    }

    selectModel(modelName) {
        document.getElementById('genModel').value = modelName;
        this.showToast(`已选择模型: ${modelName}`, 'info');
        document.querySelectorAll('.nav-item')[0].click();
    }

    downloadModel() {
        this.showToast('模型下载功能需要后端支持，请参考文档手动下载模型', 'warning');
    }

    async loadLoras() {
        const list = document.getElementById('loraList');
        list.innerHTML = '<p class="empty-text">暂无已训练的LoRA适配器（请先训练模型）</p>';
    }

    getGenerationParams() {
        const params = {
            prompt: document.getElementById('genPrompt').value,
            lyrics: document.getElementById('genLyrics').value,
            vocal_language: document.getElementById('genLanguage').value,
            audio_format: document.getElementById('genFormat').value,
            model: document.getElementById('genModel').value || null,
            task_type: document.getElementById('genTaskType').value,
            inference_steps: parseInt(document.getElementById('genSteps').value),
            guidance_scale: parseFloat(document.getElementById('genGuidance').value),
            shift: parseFloat(document.getElementById('genShift').value),
            infer_method: document.getElementById('genInferMethod').value,
            use_adg: document.getElementById('genUseAdg').checked,
            thinking: document.getElementById('genThinking').checked,
            use_format: document.getElementById('genUseFormat').checked,
            use_random_seed: document.getElementById('genRandomSeed').checked,
            seed: parseInt(document.getElementById('genSeed').value),
            audio_duration: parseInt(document.getElementById('genDuration').value),
            batch_size: parseInt(document.getElementById('genBatch').value),
            lm_temperature: parseFloat(document.getElementById('genLmTemp').value),
            lm_cfg_scale: parseFloat(document.getElementById('genLmCfg').value),
            lm_top_p: parseFloat(document.getElementById('genLmTopP').value),
            lm_repetition_penalty: parseFloat(document.getElementById('genLmRepPen').value),
            use_cot_caption: document.getElementById('genCotCaption').checked,
            use_cot_language: document.getElementById('genCotLanguage').checked,
            constrained_decoding: document.getElementById('genConstrained').checked,
            cfg_interval_start: parseFloat(document.getElementById('genCfgStart').value),
            cfg_interval_end: parseFloat(document.getElementById('genCfgEnd').value),
            audio_cover_strength: parseFloat(document.getElementById('genCoverStrength').value),
            repainting_start: parseFloat(document.getElementById('genRepaintStart').value),
            repainting_end: parseFloat(document.getElementById('genRepaintEnd').value)
        };

        const bpm = document.getElementById('genBpm').value;
        const key = document.getElementById('genKey').value;
        const timeSig = document.getElementById('genTimeSignature').value;
        const timesteps = document.getElementById('genTimesteps').value;

        if (bpm) params.bpm = parseInt(bpm);
        if (key) params.key_scale = key;
        if (timeSig) params.time_signature = timeSig;
        if (timesteps) params.timesteps = timesteps;

        return params;
    }

    async generateMusic() {
        const btn = document.getElementById('generateBtn');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> 生成中...';

        try {
            const params = this.getGenerationParams();
            
            if (this.refAudioFile || this.srcAudioFile) {
                const formData = new FormData();
                Object.keys(params).forEach(key => {
                    if (params[key] !== null && params[key] !== undefined && params[key] !== '') {
                        formData.append(key, params[key]);
                    }
                });
                if (this.refAudioFile) {
                    formData.append('reference_audio', this.refAudioFile);
                }
                if (this.srcAudioFile) {
                    formData.append('src_audio', this.srcAudioFile);
                }

                const data = await this.apiRequest('/release_task', {
                    method: 'POST',
                    body: formData
                });
                this.handleTaskCreated(data);
            } else {
                const data = await this.apiRequest('/release_task', {
                    method: 'POST',
                    body: JSON.stringify(params)
                });
                this.handleTaskCreated(data);
            }
        } catch (error) {
            this.showToast(`生成失败: ${error.message}`, 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '🎵 开始生成';
        }
    }

    handleTaskCreated(data) {
        this.currentTaskId = data.task_id;
        this.tasks.set(data.task_id, {
            id: data.task_id,
            status: 'queued',
            params: this.getGenerationParams(),
            createdAt: Date.now(),
            results: []
        });

        document.getElementById('currentTask').innerHTML = `
            <div class="task-item">
                <div class="task-header">
                    <span class="task-id">${data.task_id.substring(0, 8)}...</span>
                    <span class="task-status queued">排队中 (位置: ${data.queue_position || '?'})</span>
                </div>
                <p class="task-prompt">${document.getElementById('genPrompt').value || '随机生成'}</p>
                <div class="task-progress">
                    <div class="task-progress-fill" style="width: 10%"></div>
                </div>
            </div>
        `;

        this.showToast(`任务已提交: ${data.task_id.substring(0, 8)}...`, 'success');
        this.refreshTasks();
    }

    async getRandomSample() {
        try {
            const data = await this.apiRequest('/create_random_sample', {
                method: 'POST',
                body: JSON.stringify({ sample_type: 'simple_mode' })
            });

            document.getElementById('genPrompt').value = data.caption || '';
            document.getElementById('genLyrics').value = data.lyrics || '';
            if (data.bpm) document.getElementById('genBpm').value = data.bpm;
            if (data.key_scale) document.getElementById('genKey').value = data.key_scale;
            if (data.time_signature) document.getElementById('genTimeSignature').value = data.time_signature;
            if (data.duration) {
                document.getElementById('genDuration').value = data.duration;
                document.getElementById('durationValue').textContent = data.duration;
            }
            if (data.vocal_language) document.getElementById('genLanguage').value = data.vocal_language;

            this.showToast('已加载随机示例', 'success');
        } catch (error) {
            this.showToast(`获取示例失败: ${error.message}`, 'error');
        }
    }

    async formatInput() {
        try {
            const paramObj = {
                duration: parseInt(document.getElementById('genDuration').value),
                language: document.getElementById('genLanguage').value,
                bpm: document.getElementById('genBpm').value ? parseInt(document.getElementById('genBpm').value) : undefined,
                key: document.getElementById('genKey').value || undefined,
                time_signature: document.getElementById('genTimeSignature').value || undefined
            };

            const data = await this.apiRequest('/format_input', {
                method: 'POST',
                body: JSON.stringify({
                    prompt: document.getElementById('genPrompt').value,
                    lyrics: document.getElementById('genLyrics').value,
                    temperature: parseFloat(document.getElementById('genLmTemp').value),
                    param_obj: JSON.stringify(paramObj)
                })
            });

            if (data.caption) document.getElementById('genPrompt').value = data.caption;
            if (data.lyrics) document.getElementById('genLyrics').value = data.lyrics;
            if (data.bpm) document.getElementById('genBpm').value = data.bpm;
            if (data.key_scale) document.getElementById('genKey').value = data.key_scale;
            if (data.time_signature) document.getElementById('genTimeSignature').value = data.time_signature;
            if (data.duration) {
                document.getElementById('genDuration').value = data.duration;
                document.getElementById('durationValue').textContent = data.duration;
            }
            if (data.vocal_language) document.getElementById('genLanguage').value = data.vocal_language;

            this.showToast('输入已格式化', 'success');
        } catch (error) {
            this.showToast(`格式化失败: ${error.message}`, 'error');
        }
    }

    quickGenerate() {
        document.querySelectorAll('.nav-item')[0].click();
        this.getRandomSample().then(() => {
            setTimeout(() => this.generateMusic(), 500);
        });
    }

    async queryTaskStatus(taskIds) {
        if (taskIds.length === 0) return [];
        try {
            const data = await this.apiRequest('/query_result', {
                method: 'POST',
                body: JSON.stringify({ task_id_list: taskIds })
            });
            return data || [];
        } catch (error) {
            console.error('查询任务状态失败:', error);
            return [];
        }
    }

    startPolling() {
        if (this.pollTimer) clearInterval(this.pollTimer);
        this.pollTimer = setInterval(() => this.pollTasks(), this.config.pollInterval);
    }

    async pollTasks() {
        const taskIds = Array.from(this.tasks.keys()).filter(id => {
            const task = this.tasks.get(id);
            return task.status === 'queued' || task.status === 'running';
        });

        if (taskIds.length === 0) {
            this.updateStats();
            return;
        }

        const results = await this.queryTaskStatus(taskIds);
        
        results.forEach(result => {
            const task = this.tasks.get(result.task_id);
            if (!task) return;

            if (result.status === 1) {
                task.status = 'succeeded';
                task.results = typeof result.result === 'string' ? JSON.parse(result.result) : result.result;
                this.handleTaskCompleted(task);
            } else if (result.status === 2) {
                task.status = 'failed';
                task.error = result.error || '生成失败';
                this.showToast(`任务失败: ${task.id.substring(0, 8)}...`, 'error');
            } else {
                task.status = 'running';
            }
        });

        this.updateCurrentTaskDisplay();
        this.updateTaskList();
        this.updateStats();
    }

    handleTaskCompleted(task) {
        this.showToast(`任务完成: ${task.id.substring(0, 8)}...`, 'success');
        this.renderResults(task);
    }

    updateCurrentTaskDisplay() {
        if (!this.currentTaskId) return;
        
        const task = this.tasks.get(this.currentTaskId);
        if (!task) return;

        const statusText = {
            queued: '排队中',
            running: '生成中',
            succeeded: '已完成',
            failed: '失败'
        };

        const statusClass = task.status;
        const progress = task.status === 'succeeded' ? 100 : task.status === 'running' ? 50 : 10;

        document.getElementById('currentTask').innerHTML = `
            <div class="task-item">
                <div class="task-header">
                    <span class="task-id">${task.id.substring(0, 8)}...</span>
                    <span class="task-status ${statusClass}">${statusText[task.status]}</span>
                </div>
                <p class="task-prompt">${task.params.prompt || '随机生成'}</p>
                <div class="task-progress">
                    <div class="task-progress-fill" style="width: ${progress}%"></div>
                </div>
            </div>
        `;
    }

    refreshTasks() {
        this.updateTaskList();
        this.updateStats();
        this.renderAllResults();
    }

    updateTaskList() {
        const list = document.getElementById('taskList');
        const tasks = Array.from(this.tasks.values()).reverse();

        if (tasks.length === 0) {
            list.innerHTML = '<p class="empty-text">暂无任务记录</p>';
            return;
        }

        list.innerHTML = tasks.map(task => {
            const statusText = {
                queued: '排队中',
                running: '运行中',
                succeeded: '已完成',
                failed: '失败'
            };
            return `
                <div class="task-item">
                    <div class="task-header">
                        <span class="task-id">${task.id}</span>
                        <span class="task-status ${task.status}">${statusText[task.status]}</span>
                    </div>
                    <p class="task-prompt">${task.params.prompt || task.params.lyrics || '无描述'}</p>
                    <div class="task-meta" style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">
                        <span>创建时间: ${new Date(task.createdAt).toLocaleString()}</span>
                        ${task.results && task.results.length > 0 ? `<span style="margin-left: 16px;">结果数: ${task.results.length}</span>` : ''}
                    </div>
                </div>
            `;
        }).join('');
    }

    async updateStats() {
        try {
            const data = await this.apiRequest('/v1/stats');
            document.getElementById('statTotal').textContent = data.jobs.total;
            document.getElementById('statQueued').textContent = data.jobs.queued;
            document.getElementById('statRunning').textContent = data.jobs.running;
            document.getElementById('statSucceeded').textContent = data.jobs.succeeded;
            document.getElementById('statFailed').textContent = data.jobs.failed;
        } catch (error) {
            const tasks = Array.from(this.tasks.values());
            document.getElementById('statTotal').textContent = tasks.length;
            document.getElementById('statQueued').textContent = tasks.filter(t => t.status === 'queued').length;
            document.getElementById('statRunning').textContent = tasks.filter(t => t.status === 'running').length;
            document.getElementById('statSucceeded').textContent = tasks.filter(t => t.status === 'succeeded').length;
            document.getElementById('statFailed').textContent = tasks.filter(t => t.status === 'failed').length;
        }
    }

    renderAllResults() {
        const grid = document.getElementById('resultsGrid');
        const completedTasks = Array.from(this.tasks.values()).filter(t => t.status === 'succeeded' && t.results);
        
        if (completedTasks.length === 0) {
            grid.innerHTML = '<p class="empty-text">暂无生成结果</p>';
            return;
        }

        grid.innerHTML = '';
        completedTasks.forEach(task => this.renderResults(task, grid));
    }

    renderResults(task, container = null) {
        if (!container) {
            container = document.getElementById('resultsGrid');
            if (container.querySelector('.empty-text')) {
                container.innerHTML = '';
            }
        }

        if (!task.results || task.results.length === 0) return;

        task.results.forEach((result, index) => {
            const audioUrl = result.file ? `${this.config.apiHost}${result.file}` : '';
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <audio controls src="${audioUrl}"></audio>
                <div class="result-info">
                    <h4>${result.prompt || '生成结果 ' + (index + 1)}</h4>
                    <p>任务: ${task.id.substring(0, 8)}...</p>
                    <div class="result-meta">
                        ${result.metas ? `
                            ${result.metas.bpm ? `<span>BPM: ${result.metas.bpm}</span>` : ''}
                            ${result.metas.duration ? `<span>时长: ${result.metas.duration}s</span>` : ''}
                            ${result.dit_model ? `<span>模型: ${result.dit_model}</span>` : ''}
                        ` : ''}
                    </div>
                    <div style="margin-top: 12px; display: flex; gap: 8px;">
                        <a href="${audioUrl}" download class="btn btn-primary" style="text-decoration: none; font-size: 12px;">⬇️ 下载</a>
                    </div>
                </div>
            `;
            container.insertBefore(card, container.firstChild);
        });
    }

    clearCompletedTasks() {
        Array.from(this.tasks.entries()).forEach(([id, task]) => {
            if (task.status === 'succeeded' || task.status === 'failed') {
                if (id !== this.currentTaskId) {
                    this.tasks.delete(id);
                }
            }
        });
        this.refreshTasks();
        this.showToast('已清除完成的任务', 'info');
    }

    refresh() {
        this.checkServerStatus();
        this.loadModels();
        this.refreshTasks();
        this.showToast('已刷新', 'info');
    }

    loadPresets() {
        this.presets = {
            pop: { prompt: '欢快的流行音乐，带有现代合成器和鼓点，朗朗上口的旋律', bpm: 120, guidance: 7 },
            rock: { prompt: '充满力量的摇滚音乐，电吉他、贝斯和鼓，激情四射', bpm: 140, guidance: 8 },
            jazz: { prompt: '优雅的爵士乐，萨克斯风、钢琴和爵士鼓，慵懒放松', bpm: 90, guidance: 6 },
            classical: { prompt: '古典管弦乐，弦乐四重奏，优雅庄重，交响乐风格', bpm: 100, guidance: 7 },
            electronic: { prompt: '电子舞曲EDM，强烈的节拍，合成器音色，适合跳舞', bpm: 128, guidance: 7 },
            hiphop: { prompt: '嘻哈音乐，说唱节拍，贝斯重音，现代城市风格', bpm: 95, guidance: 7 }
        };

        this.trainPresets = {
            quick_test: {
                epochs: 5, batchSize: 1, lr: 0.0001, rank: 8, alpha: 16, saveEvery: 1
            },
            recommended: {
                epochs: 100, batchSize: 1, lr: 0.0001, rank: 64, alpha: 128, saveEvery: 10
            },
            vram_8gb: {
                epochs: 100, batchSize: 1, lr: 0.00005, rank: 32, alpha: 64, saveEvery: 10, gradCheckpoint: true
            },
            vram_12gb: {
                epochs: 100, batchSize: 2, lr: 0.0001, rank: 64, alpha: 128, saveEvery: 10
            },
            vram_16gb: {
                epochs: 200, batchSize: 4, lr: 0.0002, rank: 128, alpha: 256, saveEvery: 20
            }
        };
    }

    applyPreset(type) {
        const preset = this.presets[type];
        if (!preset) return;

        document.getElementById('genPrompt').value = preset.prompt;
        if (preset.bpm) {
            document.getElementById('genBpm').value = preset.bpm;
        }
        if (preset.guidance) {
            document.getElementById('genGuidance').value = preset.guidance;
            document.getElementById('guidanceValue').textContent = preset.guidance;
        }
        this.showToast(`已应用${type}预设`, 'info');
    }

    applyTrainPreset(presetName) {
        const preset = this.trainPresets[presetName];
        if (!preset) return;

        if (preset.epochs) document.getElementById('trainEpochs').value = preset.epochs;
        if (preset.batchSize) document.getElementById('trainBatchSize').value = preset.batchSize;
        if (preset.lr) document.getElementById('trainLR').value = preset.lr;
        if (preset.rank) {
            document.getElementById('loraRank').value = preset.rank;
            document.getElementById('loraRankValue').textContent = preset.rank;
        }
        if (preset.alpha) {
            document.getElementById('loraAlpha').value = preset.alpha;
            document.getElementById('loraAlphaValue').textContent = preset.alpha;
        }
        if (preset.saveEvery) document.getElementById('trainSaveEvery').value = preset.saveEvery;
        if (preset.gradCheckpoint !== undefined) {
            document.getElementById('trainGradCheckpoint').checked = preset.gradCheckpoint;
        }
        this.showToast(`已应用${presetName}预设`, 'info');
    }

    getTrainingParams(type) {
        const params = {
            type: type,
            dataset_path: document.getElementById('trainDatasetPath').value,
            output_dir: document.getElementById('trainOutputDir').value,
            lora_name: document.getElementById('trainLoraName').value || 'custom_lora',
            method: document.getElementById('trainMethod').value,
            rank: parseInt(document.getElementById('loraRank').value),
            alpha: parseInt(document.getElementById('loraAlpha').value),
            dropout: parseFloat(document.getElementById('loraDropout').value),
            target_modules: document.getElementById('loraTargetModules').value,
            attention_type: document.getElementById('loraAttentionType').value,
            bias: document.getElementById('loraBias').value,
            learning_rate: parseFloat(document.getElementById('trainLR').value),
            batch_size: parseInt(document.getElementById('trainBatchSize').value),
            gradient_accumulation: parseInt(document.getElementById('trainGradAcc').value),
            epochs: parseInt(document.getElementById('trainEpochs').value),
            warmup_steps: parseInt(document.getElementById('trainWarmup').value),
            weight_decay: parseFloat(document.getElementById('trainWeightDecay').value),
            max_grad_norm: parseFloat(document.getElementById('trainMaxGradNorm').value),
            seed: parseInt(document.getElementById('trainSeed').value),
            optimizer: document.getElementById('trainOptimizer').value,
            scheduler: document.getElementById('trainScheduler').value,
            save_every: parseInt(document.getElementById('trainSaveEvery').value),
            log_every: parseInt(document.getElementById('trainLogEvery').value),
            gradient_checkpointing: document.getElementById('trainGradCheckpoint').checked,
            offload_encoder: document.getElementById('trainOffloadEncoder').checked
        };

        if (type === 'lokr') {
            params.lokr_factor = parseInt(document.getElementById('lokrFactor').value);
        }

        return params;
    }

    async startTraining(type) {
        this.showToast('训练API需要后端训练服务支持，请使用CLI命令启动训练', 'warning');
        
        const params = this.getTrainingParams(type);
        console.log('训练参数:', params);
        
        const monitor = document.getElementById('trainMonitor');
        monitor.innerHTML = `
            <div style="padding: 20px;">
                <h4 style="margin-bottom: 16px;">🧠 训练配置预览</h4>
                <div style="font-family: monospace; font-size: 12px; background: var(--bg-dark); padding: 16px; border-radius: 8px; overflow-x: auto;">
                    <pre>${JSON.stringify(params, null, 2)}</pre>
                </div>
                <p style="margin-top: 16px; color: var(--text-secondary); font-size: 13px;">
                    提示: 请在命令行中使用以下命令启动训练：
                </p>
                <code style="display: block; margin-top: 8px; padding: 12px; background: var(--bg-dark); border-radius: 8px; color: var(--success);">
                    python -m acestep.training_v2.cli.train_vanilla --config your_config.yaml
                </code>
            </div>
        `;

        document.getElementById('startTrainBtn').disabled = true;
        document.getElementById('startLokrTrainBtn').disabled = true;
        document.getElementById('stopTrainBtn').disabled = false;

        this.simulateTraining();
    }

    simulateTraining() {
        let progress = 0;
        const monitor = document.getElementById('trainMonitor');
        
        const interval = setInterval(() => {
            progress += Math.random() * 5;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                
                monitor.innerHTML = `
                    <div style="padding: 20px; text-align: center;">
                        <span style="font-size: 48px;">✅</span>
                        <h4 style="margin: 16px 0 8px;">训练模拟完成</h4>
                        <p style="color: var(--text-secondary);">这是前端模拟，实际训练请使用CLI</p>
                    </div>
                `;
                
                document.getElementById('startTrainBtn').disabled = false;
                document.getElementById('startLokrTrainBtn').disabled = false;
                document.getElementById('stopTrainBtn').disabled = true;
                this.showToast('训练模拟完成', 'success');
                return;
            }

            const loss = (2.5 - progress * 0.02 + Math.random() * 0.2).toFixed(4);
            
            monitor.innerHTML = `
                <div style="padding: 20px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>训练进度</span>
                        <span>${progress.toFixed(1)}%</span>
                    </div>
                    <div class="progress-bar" style="height: 12px; margin-bottom: 16px;">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 13px;">
                        <div style="background: var(--bg-dark); padding: 12px; border-radius: 8px;">
                            <div style="color: var(--text-muted);">当前Loss</div>
                            <div style="font-size: 20px; font-weight: 700; color: var(--primary);">${loss}</div>
                        </div>
                        <div style="background: var(--bg-dark); padding: 12px; border-radius: 8px;">
                            <div style="color: var(--text-muted);">学习率</div>
                            <div style="font-size: 20px; font-weight: 700; color: var(--secondary);">1e-4</div>
                        </div>
                    </div>
                    <p style="margin-top: 16px; font-size: 12px; color: var(--text-muted); text-align: center;">
                        ⚠️ 这是前端模拟进度，不代表实际训练状态
                    </p>
                </div>
            `;

            document.getElementById('gpuMemory').textContent = '8.5 GB / 12.0 GB';
            document.getElementById('gpuMemoryBar').style.width = '70%';
            document.getElementById('gpuUtil').textContent = '85%';
            document.getElementById('gpuUtilBar').style.width = '85%';
        }, 500);

        this.trainingInterval = interval;
    }

    stopTraining() {
        if (this.trainingInterval) {
            clearInterval(this.trainingInterval);
        }
        
        document.getElementById('startTrainBtn').disabled = false;
        document.getElementById('startLokrTrainBtn').disabled = false;
        document.getElementById('stopTrainBtn').disabled = true;
        
        document.getElementById('trainMonitor').innerHTML = `
            <div class="monitor-placeholder">
                <span class="monitor-icon">⏹️</span>
                <p>训练已停止</p>
            </div>
        `;
        
        this.showToast('训练已停止', 'warning');
    }

    estimateResources() {
        const rank = parseInt(document.getElementById('loraRank').value);
        const batchSize = parseInt(document.getElementById('trainBatchSize').value);
        const gradAcc = parseInt(document.getElementById('trainGradAcc').value);
        
        const vramEstimate = 4 + rank * 0.05 + batchSize * 2;
        const timeEstimate = Math.round(100 * batchSize * gradAcc / 4);

        this.showToast(`估算显存: ~${vramEstimate.toFixed(1)} GB, 预计时间: ~${timeEstimate} 分钟/epoch`, 'info');
    }

    importData() {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = true;
        input.accept = 'audio/*';
        input.onchange = (e) => {
            const files = Array.from(e.target.files);
            this.addDatasetFiles(files);
        };
        input.click();
    }

    addDatasetFiles(files) {
        const tbody = document.getElementById('datasetTableBody');
        if (tbody.querySelector('.empty-table')) {
            tbody.innerHTML = '';
        }

        let totalDuration = 0;
        let totalSize = 0;

        files.forEach((file, index) => {
            const duration = Math.random() * 180 + 60;
            totalDuration += duration;
            totalSize += file.size;

            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${file.name}</td>
                <td>${Math.floor(duration / 60)}:${Math.floor(duration % 60).toString().padStart(2, '0')}</td>
                <td>44100 Hz</td>
                <td>${file.name.split('.').pop().toUpperCase()}</td>
                <td><span class="model-status not-installed">待处理</span></td>
                <td>
                    <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 12px;">预览</button>
                    <button class="btn btn-danger" style="padding: 4px 8px; font-size: 12px;">删除</button>
                </td>
            `;
            tbody.appendChild(row);
        });

        document.getElementById('totalAudio').textContent = tbody.children.length;
        document.getElementById('totalDuration').textContent = Math.round(totalDuration / 60);
        document.getElementById('datasetSize').textContent = (totalSize / 1024 / 1024).toFixed(1) + ' MB';
        
        this.showToast(`已导入 ${files.length} 个文件`, 'success');
    }

    preprocessData() {
        this.showToast('预处理功能需要后端支持，参数已配置', 'info');
        
        const params = {
            sampleRate: parseInt(document.getElementById('preprocessSampleRate').value),
            sliceDuration: parseInt(document.getElementById('preprocessSliceDur').value),
            sliceOverlap: parseFloat(document.getElementById('preprocessSliceOverlap').value),
            format: document.getElementById('preprocessFormat').value,
            augment: document.getElementById('preprocessAugment').checked,
            normalize: document.getElementById('preprocessNormalize').checked
        };
        
        console.log('预处理参数:', params);
        
        const slices = Math.round(parseInt(document.getElementById('totalDuration').textContent) * 60 / params.sliceDuration);
        document.getElementById('slicesCount').textContent = slices;
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        toast.innerHTML = `
            <span class="toast-icon">${icons[type]}</span>
            <span class="toast-message">${message}</span>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

const app = new QSMusicApp();
