// 全局变量
let allFaceRecords = [];
let currentPage = 1;
let recordsPerPage = 30;
let totalPages = 1;
let detectionSettings = {
    detection_threshold: 0.5,  // 降低检测阈值以提高召回率
    face_padding: 5,          // 减少padding以减轻处理负担
    similarity_threshold: 0.6, // 提高相似度阈值使匹配更严格
    vote_threshold: 2,        // 降低投票阈值
    enable_preprocessing: false, // 默认关闭预处理以提高性能
    enable_voting: false,     // 默认关闭投票机制以提高性能
    use_large_model: false,   // 默认使用小模型以提高性能
    enable_emotion_detection: false,
    enable_fatigue_detection: false,
    frame_skip: 0
};
let refreshInterval = null; // 用于存储定时刷新的interval ID

// 自动更新定时器
let autoUpdateTimer = null;

// 开始自动更新
function startAutoUpdate() {
    if (autoUpdateTimer) {
        clearInterval(autoUpdateTimer);
    }
    autoUpdateTimer = setInterval(refreshData, 30000); // 每30秒更新一次
    console.log("自动更新已启动");
}

// 停止自动更新
function stopAutoUpdate() {
    if (autoUpdateTimer) {
        clearInterval(autoUpdateTimer);
        autoUpdateTimer = null;
    }
    console.log("自动更新已停止");
}

// 刷新所有数据
async function refreshData() {
    try {
        // 显示加载动画
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 更新中...';
        }

        // 并行加载所有数据
        await Promise.all([
            loadDetectionStats(),
            loadPersonStatsData(),
            loadRecentDetections()
        ]);

        // 更新成功提示
        if (refreshBtn) {
            refreshBtn.innerHTML = '<i class="fas fa-sync"></i> 刷新';
            refreshBtn.disabled = false;
        }
        
        showToast('数据更新成功', 'success');
    } catch (error) {
        console.error('更新数据失败:', error);
        showToast('更新数据失败: ' + error.message, 'error');
        
        // 恢复刷新按钮状态
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.innerHTML = '<i class="fas fa-sync"></i> 刷新';
            refreshBtn.disabled = false;
        }
    }
}

// 加载检测统计信息
async function loadDetectionStats() {
    try {
        const response = await fetch('/detection_stats');
        const data = await response.json();
        
        if (data.success) {
            // 更新当前检测到的人脸数
            const currentFacesElement = document.getElementById('current-faces');
            if (currentFacesElement) {
                currentFacesElement.textContent = data.stats.current_faces;
            }
            
            // 更新总检测记录数
            const totalRecordsElement = document.getElementById('total-records');
            if (totalRecordsElement) {
                totalRecordsElement.textContent = data.stats.total_records;
            }
        }
    } catch (error) {
        console.error('加载检测统计信息失败:', error);
        throw error;
    }
}

// 加载人员统计数据
async function loadPersonStatsData() {
    try {
        const response = await fetch('/person_stats');
        const data = await response.json();
        
        if (data.success) {
            // 更新人员统计表格
            updatePersonStatsTable(data.stats);
            
            // 更新情感趋势图
            drawEmotionTrendChart(data.stats.emotion_trend);
            
            // 更新疲劳趋势图
            drawFatigueChart(data.stats.fatigue_trend);
        }
    } catch (error) {
        console.error('加载人员统计数据失败:', error);
        throw error;
    }
}

// 加载最近检测记录
async function loadRecentDetections() {
    try {
        const response = await fetch('/recent_detections');
        const data = await response.json();
        
        if (data.success) {
            updateRecentDetectionsTable(data.detections);
        }
    } catch (error) {
        console.error('加载最近检测记录失败:', error);
        throw error;
    }
}

// 页面加载完成后启动自动更新
document.addEventListener('DOMContentLoaded', function() {
    // 设置刷新按钮点击事件
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshData);
    }
    
    // 启动自动更新
    startAutoUpdate();
    
    // 初始加载数据
    refreshData();
});

// 页面卸载前停止自动更新
window.addEventListener('beforeunload', function() {
    stopAutoUpdate();
});

// 修改刷新按钮点击事件
function refreshData() {
    // 显示加载动画
    showToast('info', '正在刷新数据...');
    
    // 立即更新所有数据
    Promise.all([
        loadDetectionStats(),
        loadFaceRecords(),
        loadPersonStatsData()
    ]).then(() => {
        showToast('success', '数据刷新完成');
    }).catch(error => {
        showToast('error', '刷新数据失败: ' + error.message);
    });
}

// 页面加载时的主函数
window.onload = function() {
    console.log('页面加载完成，开始初始化');
    
    try {
        // 初始化功能模块
        checkLoginStatus();
        initTimeDisplay();
        initNavigation();
        initDetectionButtons();
        initFaceManagement();
        initSettingsPanel();
        setupSliders();
        loadDetectionSettings();
        setupEventDelegation();
        initPagination();
        
        // 加载初始数据
        loadDetectionStats();
        loadFaceRecords();
        loadFaces();
        
        // 添加手动刷新按钮
        addRefreshButton();
        
        // 启动数据自动刷新（每分钟一次）
        startDataRefresh();

        console.log('初始化完成');
    } catch (e) {
        console.error('初始化错误:', e);
    }
};

// 添加手动刷新按钮
function addRefreshButton() {
    console.log('添加手动刷新按钮');
    
    // 检查是否已经存在刷新按钮，避免重复添加
    if (document.getElementById('manual-refresh-btn')) {
        console.log('刷新按钮已存在，不再重复添加');
        return;
    }
    
    // 在统计卡片区域添加刷新按钮
    const statsCardRow = document.querySelector('.stats-card').closest('.row');
    if (statsCardRow) {
        // 创建刷新按钮容器
        const refreshBtnDiv = document.createElement('div');
        refreshBtnDiv.className = 'col-12 text-end mb-2';
        refreshBtnDiv.innerHTML = `
            <button id="manual-refresh-btn" class="btn btn-outline-primary btn-sm">
                <i class="fas fa-sync-alt"></i> 刷新数据
            </button>
        `;
        
        // 插入到卡片区域前面
        statsCardRow.parentNode.insertBefore(refreshBtnDiv, statsCardRow);
        
        // 添加点击事件
        document.getElementById('manual-refresh-btn').addEventListener('click', function() {
            this.disabled = true;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 刷新中...';
            
            // 执行刷新
            Promise.all([
                loadDetectionStats(),
                loadFaceRecords(),
                loadFaces()
            ]).then(() => {
                showToast('success', '数据已刷新');
                this.disabled = false;
                this.innerHTML = '<i class="fas fa-sync-alt"></i> 刷新数据';
            }).catch(error => {
                console.error('手动刷新错误:', error);
                showToast('error', '刷新失败，请重试');
                this.disabled = false;
                this.innerHTML = '<i class="fas fa-sync-alt"></i> 刷新数据';
            });
        });
    }
}

// 修改数据自动刷新功能为每分钟一次
function startDataRefresh() {
    console.log('启动数据自动刷新（每分钟一次）');
    // 清除可能已存在的interval
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    
    // 设置新的刷新interval，每60秒（1分钟）刷新一次
    refreshInterval = setInterval(() => {
        console.log('自动刷新数据...');
        loadDetectionStats();
        
        // 判断是否在主页，如果是则更新最近记录
        if (document.getElementById('home-page').classList.contains('active')) {
            updateRecentActivityFromStats();
        }
        
        // 如果在数据库页面，定期刷新记录
        if (document.getElementById('database-page').classList.contains('active')) {
            loadFaceRecords();
            loadFaces();
        }
    }, 60000); // 每60秒（1分钟）刷新一次
    
    console.log('数据自动刷新已启动（每分钟一次）');
}

// 停止数据刷新
function stopDataRefresh() {
    console.log('停止数据自动刷新');
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

// 从检测统计更新最近活动记录
function updateRecentActivityFromStats() {
    fetch('/detection_stats')
        .then(response => response.json())
        .then(data => {
            console.log('获取到的检测统计:', data);
            
            // 更新当前检测到的人脸数
            document.getElementById('current-faces-count').textContent = data.total_faces || 0;
            
            // 更新最后检测时间
            if (data.timestamp) {
                document.getElementById('detection-timestamp').textContent = `上次更新: ${data.timestamp}`;
            }
            
            // 更新总检测记录数
            const recordCount = data.last_10_records ? data.last_10_records.length : 0;
            document.getElementById('total-records').textContent = recordCount;
            
            // 更新最近活动记录
            if (data.last_10_records && data.last_10_records.length > 0) {
                updateRecentActivity(data.last_10_records);
            }
        })
        .catch(error => {
            console.error('获取检测统计错误:', error);
        });
}

// 修改loadDetectionStats函数以更新总记录数
function loadDetectionStats() {
    fetch('/detection_stats')
        .then(response => response.json())
        .then(data => {
            console.log('获取到的检测统计:', data);
            
            // 更新当前检测到的人脸数
            document.getElementById('current-faces-count').textContent = data.total_faces || 0;
            
            // 更新最后检测时间
            if (data.timestamp) {
                document.getElementById('detection-timestamp').textContent = `上次更新: ${data.timestamp}`;
            }
            
            // 获取总记录数
            fetch('/face_records')
                .then(response => response.json())
                .then(records => {
                    document.getElementById('total-records').textContent = records.length;
                })
                .catch(error => {
                    console.error('获取人脸记录错误:', error);
                });
            
            // 更新最近活动记录
            if (data.last_10_records && data.last_10_records.length > 0) {
                updateRecentActivity(data.last_10_records);
            }
        })
        .catch(error => {
            console.error('获取检测统计错误:', error);
        });
}

// 修改startDetection函数，启动检测时开始数据刷新
async function startDetection() {
    console.log('启动人脸检测');
    
    try {
        // 显示加载模态框
        const cameraModal = new bootstrap.Modal(document.getElementById('cameraStartModal'));
        cameraModal.show();
        
        // 模拟进度条加载
        const progressBar = document.getElementById('camera-load-progress');
        const statusText = document.getElementById('camera-status-text');
        
        // 重置进度条
        progressBar.style.width = '0%';
        
        // 更新状态文本和进度条
        function updateProgress(percent, text) {
            progressBar.style.width = `${percent}%`;
            statusText.textContent = text;
        }
        
        // 进度动画
        updateProgress(10, '正在连接摄像头...');
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // 发送启动请求到服务器
        updateProgress(30, '正在初始化摄像头...');
        const response = await fetch('/start_detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                camera_index: 0  // 默认使用摄像头0
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateProgress(60, '摄像头已就绪，正在加载模型...');
            await new Promise(resolve => setTimeout(resolve, 500));
            
            updateProgress(80, '正在启动检测服务...');
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // 更新视频源
            const videoStream = document.getElementById('video-stream');
            const streamOverlay = document.getElementById('stream-overlay');
            
            // 隐藏覆盖层，显示视频流
            streamOverlay.style.display = 'none';
            videoStream.src = '/video_feed';
            
            // 更新UI状态
            updateDetectionUIState(true);
            
            // 更新系统状态卡片
            document.getElementById('system-status').textContent = '运行中';
            document.getElementById('system-status').style.color = '#34a853';
            document.getElementById('system-message').textContent = '人脸检测系统正在运行中';
            
            updateProgress(100, '检测系统已成功启动！');
            await new Promise(resolve => setTimeout(resolve, 800));
            
            // 隐藏模态框
            cameraModal.hide();
            
            // 启动数据自动刷新
            startDataRefresh();
            
            showToast('success', '人脸检测已成功启动');
        } else {
            updateProgress(100, `启动失败: ${data.message}`);
            showToast('error', `启动失败: ${data.message}`);
            
            // 隐藏模态框
            setTimeout(() => {
                cameraModal.hide();
            }, 2000);
        }
    } catch (error) {
        console.error('启动检测错误:', error);
        showToast('error', '启动检测失败，请稍后再试');
    }
}

// 修改stopDetection函数，停止检测时暂停数据刷新
function stopDetection() {
    console.log('停止人脸检测');
    
    fetch('/stop_detection')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 恢复默认图像和覆盖层
                const videoStream = document.getElementById('video-stream');
                const streamOverlay = document.getElementById('stream-overlay');
                
                videoStream.src = '/static/img/placeholder.jpg';
                streamOverlay.style.display = 'flex';
                
                // 更新UI状态
                updateDetectionUIState(false);
                
                // 更新系统状态卡片
                document.getElementById('system-status').textContent = '已停止';
                document.getElementById('system-status').style.color = '#ea4335';
                document.getElementById('system-message').textContent = '检测系统已停止，点击"启动"重新开始';
                
                // 停止数据自动刷新
                stopDataRefresh();
                
                showToast('success', '人脸检测已停止');
            } else {
                showToast('error', `停止失败: ${data.message}`);
            }
        })
        .catch(error => {
            console.error('停止检测错误:', error);
            showToast('error', '停止检测失败，请稍后再试');
        });
}

// 检查登录状态
function checkLoginStatus() {
    console.log('检查登录状态');
    fetch('/check_login')
        .then(response => response.json())
        .then(data => {
            if (!data.logged_in) {
                window.location.href = '/login';
                return;
            }
            
            const usernameDisplay = document.getElementById('username-display');
            if (usernameDisplay) {
                usernameDisplay.textContent = data.username;
            }
            
            // 加载初始数据
            loadDetectionStats();
            loadFaceRecords();
            loadFaces();
        })
        .catch(error => {
            console.error('检查登录状态出错:', error);
        });
}

// 初始化时间显示
function initTimeDisplay() {
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);
}

// 更新当前时间
function updateCurrentTime() {
    const now = new Date();
    const timeElement = document.getElementById('current-time');
    
    if (timeElement) {
        timeElement.textContent = now.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    }
}

// 初始化导航功能
function initNavigation() {
    console.log('初始化导航功能');
    
    // 侧边栏折叠功能
    const sidebarCollapse = document.getElementById('sidebarCollapse');
    if (sidebarCollapse) {
        sidebarCollapse.onclick = function() {
            const sidebar = document.getElementById('sidebar');
            const content = document.getElementById('content');
            
            sidebar.classList.toggle('active');
            content.classList.toggle('active');
        };
    }
    
    // 导航链接功能
    const navLinks = document.querySelectorAll('#sidebar a[data-page]');
    navLinks.forEach(link => {
        link.onclick = function(e) {
            e.preventDefault();
            const targetPage = this.getAttribute('data-page');
            console.log('切换到页面:', targetPage);
            
            // 更新活动导航项
            document.querySelectorAll('#sidebar li').forEach(item => {
                item.classList.remove('active');
            });
            this.parentElement.classList.add('active');
            
            // 显示相应页面
            document.querySelectorAll('.page-content').forEach(page => {
                page.classList.remove('active');
            });
            document.getElementById(`${targetPage}-page`).classList.add('active');
        };
    });
    
    // 查看所有记录按钮
    const viewAllBtn = document.getElementById('view-all-btn');
    if (viewAllBtn) {
        viewAllBtn.onclick = function() {
            console.log('查看所有记录');
            // 更新活动导航项
            document.querySelectorAll('#sidebar li').forEach(item => {
                item.classList.remove('active');
            });
            const dbNavItem = document.querySelector('#sidebar li a[data-page="database"]');
            if (dbNavItem) {
                dbNavItem.parentElement.classList.add('active');
            }
            
            // 显示数据库页面
            document.querySelectorAll('.page-content').forEach(page => {
                page.classList.remove('active');
            });
            document.getElementById('database-page').classList.add('active');
        };
    }
}

// 初始化检测按钮
function initDetectionButtons() {
    console.log('初始化检测按钮');
    
    const startDetectionBtn = document.getElementById('start-detection-btn');
    const stopDetectionBtn = document.getElementById('stop-detection-btn');
    
    if (!startDetectionBtn || !stopDetectionBtn) {
        console.warn('检测按钮不存在');
        return;
    }
    
    startDetectionBtn.onclick = startDetection;
    stopDetectionBtn.onclick = stopDetection;
}

// 更新检测UI状态
function updateDetectionUIState(isRunning) {
    const startDetectionBtn = document.getElementById('start-detection-btn');
    const stopDetectionBtn = document.getElementById('stop-detection-btn');
    const videoStream = document.getElementById('video-stream');
    const streamOverlay = document.getElementById('stream-overlay');
    const systemStatus = document.getElementById('system-status');
    const systemMessage = document.getElementById('system-message');
    
    if (isRunning) {
        // 正在运行状态
        startDetectionBtn.disabled = true;
        stopDetectionBtn.disabled = false;
        streamOverlay.classList.add('hidden');
        videoStream.src = '/video_feed';
        systemStatus.textContent = '运行中';
        systemStatus.style.color = '#34a853';
        systemMessage.textContent = '检测系统已启动，正在进行人脸识别';
    } else {
        // 已停止状态
        startDetectionBtn.disabled = false;
        stopDetectionBtn.disabled = true;
        streamOverlay.classList.remove('hidden');
        videoStream.src = '/static/img/placeholder.jpg';
        systemStatus.textContent = '已停止';
        systemStatus.style.color = '#ea4335';
        systemMessage.textContent = '检测系统已停止，点击启动按钮重新开始';
    }
}

// 初始化人脸管理功能
function initFaceManagement() {
    console.log('初始化人脸管理功能');
    
    const refreshFacesBtn = document.getElementById('refresh-faces-btn');
    const clearFacesBtn = document.getElementById('clear-faces-btn');
    const confirmClearBtn = document.getElementById('confirm-clear-btn');
    
    if (refreshFacesBtn) {
        refreshFacesBtn.onclick = refreshFaces;
    }
    
    if (clearFacesBtn) {
        clearFacesBtn.onclick = function() {
            const clearDatabaseModal = new bootstrap.Modal(document.getElementById('clearDatabaseModal'));
            clearDatabaseModal.show();
        };
    }
    
    if (confirmClearBtn) {
        confirmClearBtn.onclick = clearFacesDatabase;
    }
    
    // 删除所有记录相关
    const deleteAllRecordsBtn = document.getElementById('delete-all-records-btn');
    const confirmDeleteAllRecordsBtn = document.getElementById('confirm-delete-all-records-btn');
    
    if (deleteAllRecordsBtn) {
        deleteAllRecordsBtn.onclick = function() {
            const deleteAllRecordsModal = new bootstrap.Modal(document.getElementById('deleteAllRecordsModal'));
            deleteAllRecordsModal.show();
        };
    }
    
    if (confirmDeleteAllRecordsBtn) {
        confirmDeleteAllRecordsBtn.onclick = deleteAllRecords;
    }
}

// 刷新人脸数据
function refreshFaces() {
    console.log('刷新人脸数据');
    const refreshBtn = document.getElementById('refresh-faces-btn');
    
    if (refreshBtn) {
        refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 加载中';
        refreshBtn.disabled = true;
    }
    
    loadFaces()
        .then(() => {
            showToast('success', '数据刷新成功');
        })
        .catch(() => {
            showToast('error', '数据刷新失败');
        })
        .finally(() => {
            if (refreshBtn) {
                refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> 刷新';
                refreshBtn.disabled = false;
            }
        });
}

// 清空人脸数据库
function clearFacesDatabase() {
    console.log('清空人脸数据库');
    const confirmClearBtn = document.getElementById('confirm-clear-btn');
    const clearDatabaseModal = bootstrap.Modal.getInstance(document.getElementById('clearDatabaseModal'));
    
    if (confirmClearBtn) {
        confirmClearBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 处理中';
        confirmClearBtn.disabled = true;
    }
    
    fetch('/clear_faces', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('success', '人脸数据库已成功清空');
            loadFaces();
            
            if (clearDatabaseModal) {
                clearDatabaseModal.hide();
            }
        } else {
            showToast('error', `清空数据库失败: ${data.message || '未知错误'}`);
        }
    })
    .catch(error => {
        console.error('清空数据库出错:', error);
        showToast('error', '服务器错误，请稍后重试');
    })
    .finally(() => {
        if (confirmClearBtn) {
            confirmClearBtn.innerHTML = '<i class="fas fa-trash-alt"></i> 确认清空';
            confirmClearBtn.disabled = false;
        }
    });
}

// 删除所有记录
function deleteAllRecords() {
    console.log('删除所有记录');
    const confirmDeleteAllRecordsBtn = document.getElementById('confirm-delete-all-records-btn');
    const deleteAllRecordsModal = bootstrap.Modal.getInstance(document.getElementById('deleteAllRecordsModal'));
    
    if (confirmDeleteAllRecordsBtn) {
        confirmDeleteAllRecordsBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 处理中';
        confirmDeleteAllRecordsBtn.disabled = true;
    }
    
    fetch('/delete_all_records', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('success', '所有检测记录已删除');
            loadFaceRecords();
            
            if (deleteAllRecordsModal) {
                deleteAllRecordsModal.hide();
            }
        } else {
            showToast('error', `删除失败: ${data.message || '未知错误'}`);
        }
    })
    .catch(error => {
        console.error('删除所有记录出错:', error);
        showToast('error', '服务器错误，请稍后重试');
    })
    .finally(() => {
        if (confirmDeleteAllRecordsBtn) {
            confirmDeleteAllRecordsBtn.innerHTML = '<i class="fas fa-trash-alt"></i> 确认删除';
            confirmDeleteAllRecordsBtn.disabled = false;
        }
    });
}

// 初始化设置面板
function initSettingsPanel() {
    console.log('初始化设置面板');
    
    // 滑动条事件监听
    setupSliders();
    
    // 设置按钮事件监听
    const openSettingsBtn = document.getElementById('open-settings-btn');
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    
    if (openSettingsBtn) {
        openSettingsBtn.onclick = function() {
            loadDetectionSettings();
            const settingsModal = new bootstrap.Modal(document.getElementById('settingsModal'));
            settingsModal.show();
        };
    }
    
    if (saveSettingsBtn) {
        saveSettingsBtn.onclick = saveDetectionSettings;
    }
    
    // 初始加载设置
    loadDetectionSettings();
}

// 设置滑动条事件
function setupSliders() {
    try {
        const sliders = {
            'detection-threshold': 'detection-threshold-value',
            'face-padding': 'face-padding-value',
            'similarity-threshold': 'similarity-threshold-value',
            'vote-threshold': 'vote-threshold-value',
            'frame-skip': 'frame-skip-value'
        };
        
        for (const [sliderId, valueId] of Object.entries(sliders)) {
            const slider = document.getElementById(sliderId);
            const valueDisplay = document.getElementById(valueId);
            
            if (slider && valueDisplay) {
                slider.oninput = function() {
                    valueDisplay.textContent = this.value;
                };
            }
        }
        
        // 帧跳过启用/禁用功能
        const frameSkipCheck = document.getElementById('enable-frame-skip');
        const frameSkipSlider = document.getElementById('frame-skip');
        
        if (frameSkipCheck && frameSkipSlider) {
            frameSkipCheck.onchange = function() {
                frameSkipSlider.disabled = !this.checked;
            };
        }
    } catch (e) {
        console.error('设置滑动条出错:', e);
    }
}

// 更新UI滑动条显示值
function updateSliderValues() {
    try {
        document.getElementById('detection-threshold').value = detectionSettings.detection_threshold;
        document.getElementById('face-padding').value = detectionSettings.face_padding;
        document.getElementById('similarity-threshold').value = detectionSettings.similarity_threshold;
        document.getElementById('vote-threshold').value = detectionSettings.vote_threshold;
        
        document.getElementById('detection-threshold-value').textContent = detectionSettings.detection_threshold;
        document.getElementById('face-padding-value').textContent = detectionSettings.face_padding;
        document.getElementById('similarity-threshold-value').textContent = detectionSettings.similarity_threshold;
        document.getElementById('vote-threshold-value').textContent = detectionSettings.vote_threshold;
        
        // 更新复选框
        document.getElementById('enable-preprocessing').checked = detectionSettings.enable_preprocessing;
        document.getElementById('enable-voting').checked = detectionSettings.enable_voting;
        document.getElementById('use-large-model').checked = detectionSettings.use_large_model;
    } catch (e) {
        console.error('更新设置界面出错:', e);
    }
}

// 从UI获取设置值
function getSettingsFromUI() {
    try {
        // 获取基本设置
        const settings = {
            detection_threshold: parseFloat(document.getElementById('detection-threshold').value),
            face_padding: parseInt(document.getElementById('face-padding').value),
            similarity_threshold: parseFloat(document.getElementById('similarity-threshold').value),
            vote_threshold: parseInt(document.getElementById('vote-threshold').value),
            enable_preprocessing: document.getElementById('enable-preprocessing').checked,
            enable_voting: document.getElementById('enable-voting').checked,
            use_large_model: document.getElementById('use-large-model').checked
        };
        
        // 获取增强功能设置
        const emotionDetectionEl = document.getElementById('enable-emotion-detection');
        const fatigueDetectionEl = document.getElementById('enable-fatigue-detection');
        const frameSkipEl = document.getElementById('enable-frame-skip');
        const frameSkipValueEl = document.getElementById('frame-skip');
        
        // 如果元素存在，添加到设置中
        if (emotionDetectionEl) {
            settings.enable_emotion_detection = emotionDetectionEl.checked;
        }
        
        if (fatigueDetectionEl) {
            settings.enable_fatigue_detection = fatigueDetectionEl.checked;
        }
        
        if (frameSkipEl && frameSkipValueEl) {
            // 如果跳帧功能开启，使用滑块值；否则设为0
            settings.frame_skip = frameSkipEl.checked ? parseInt(frameSkipValueEl.value) : 0;
        }
        
        return settings;
    } catch (e) {
        console.error('获取UI设置出错:', e);
        return {};
    }
}

// 从服务器加载检测设置
function loadDetectionSettings() {
    fetch('/detection_settings')
        .then(response => response.json())
        .then(data => {
            console.log('获取到的检测设置:', data);
            detectionSettings = data; // 保存到全局变量
            
            try {
                // 更新基本设置值
                updateSliderValues();
                
                // 更新增强功能设置
                const emotionDetectionEl = document.getElementById('enable-emotion-detection');
                const fatigueDetectionEl = document.getElementById('enable-fatigue-detection');
                const frameSkipEl = document.getElementById('enable-frame-skip');
                const frameSkipValueEl = document.getElementById('frame-skip');
                const frameSkipValueDisplayEl = document.getElementById('frame-skip-value');
                
                // 更新情感检测开关
                if (emotionDetectionEl && 'enable_emotion_detection' in data) {
                    emotionDetectionEl.checked = data.enable_emotion_detection;
                }
                
                // 更新疲劳检测开关
                if (fatigueDetectionEl && 'enable_fatigue_detection' in data) {
                    fatigueDetectionEl.checked = data.enable_fatigue_detection;
                }
                
                // 更新帧跳过设置
                if (frameSkipEl && frameSkipValueEl && frameSkipValueDisplayEl && 'frame_skip' in data) {
                    frameSkipEl.checked = data.frame_skip > 0;
                    frameSkipValueEl.value = data.frame_skip;
                    frameSkipValueEl.disabled = !frameSkipEl.checked;
                    frameSkipValueDisplayEl.textContent = data.frame_skip;
                }
            } catch (e) {
                console.error('更新设置UI出错:', e);
            }
        })
        .catch(error => {
            console.error('加载检测设置出错:', error);
            showToast('error', '加载设置失败');
        });
}

// 保存检测设置
function saveDetectionSettings() {
    // 获取UI中的设置值
    detectionSettings = getSettingsFromUI();
    
    // 保存到localStorage
    localStorage.setItem('detectionSettings', JSON.stringify(detectionSettings));
    
    // 发送设置到后端
    fetch('/update_detection_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(detectionSettings)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('success', '检测设置已更新');
            
            const settingsModal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
            if (settingsModal) {
                settingsModal.hide();
            }
        } else {
            alert('保存设置失败: ' + (data.message || '未知错误'));
        }
    })
    .catch(error => {
        console.error('保存设置出错:', error);
        alert('保存设置时发生错误，请稍后重试');
    });
}

// 初始化分页功能
function initPagination() {
    console.log('初始化分页功能');
    
    const prevPageBtn = document.getElementById('prev-page');
    const nextPageBtn = document.getElementById('next-page');
    const recordsPerPageSelect = document.getElementById('records-per-page');
    
    if (prevPageBtn) {
        prevPageBtn.onclick = function(e) {
            e.preventDefault();
            if (currentPage > 1) {
                goToPage(currentPage - 1);
            }
        };
    }
    
    if (nextPageBtn) {
        nextPageBtn.onclick = function(e) {
            e.preventDefault();
            if (currentPage < totalPages) {
                goToPage(currentPage + 1);
            }
        };
    }
    
    if (recordsPerPageSelect) {
        recordsPerPageSelect.onchange = function() {
            recordsPerPage = parseInt(this.value);
            currentPage = 1; // 重置为第一页
            totalPages = Math.ceil(allFaceRecords.length / recordsPerPage);
            updatePaginationInfo();
            renderCurrentPageRecords();
        };
    }
}

// 设置事件委托
function setupEventDelegation() {
    console.log('设置事件委托');
    
    // 使用单一的事件委托处理所有删除按钮
    document.addEventListener('click', function(e) {
        // 处理记录删除按钮
        const recordDeleteBtn = e.target.closest('.delete-record-btn');
        if (recordDeleteBtn) {
            handleRecordDelete(recordDeleteBtn);
            return;
        }
        
        // 处理人脸删除按钮
        const faceDeleteBtn = e.target.closest('.delete-face-btn');
        if (faceDeleteBtn) {
            handleFaceDelete(faceDeleteBtn);
            return;
        }
    });
}

// 处理记录删除
function handleRecordDelete(button) {
    const recordId = button.getAttribute('data-id');
    if (!recordId) return;
    
    if (confirm(`确定要删除ID为${recordId}的记录吗？`)) {
        deleteRecord(recordId);
    }
}

// 处理人脸删除
function handleFaceDelete(button) {
    const faceId = button.getAttribute('data-id');
    if (!faceId) return;
    
    if (confirm(`确定要删除ID为${faceId}的人脸吗？此操作不可恢复。`)) {
        deleteFace(faceId);
    }
}

// 删除人脸
function deleteFace(faceId) {
    console.log(`删除人脸ID: ${faceId}`);
    
    fetch(`/delete_face/${faceId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('success', `人脸 #${faceId} 已删除`);
            loadFaces(); // 刷新人脸列表
        } else {
            showToast('error', `删除失败: ${data.message || '未知错误'}`);
        }
    })
    .catch(error => {
        console.error('删除人脸出错:', error);
        showToast('error', '删除人脸时发生错误');
    });
}

// 删除记录
function deleteRecord(recordId) {
    console.log(`删除记录ID: ${recordId}`);
    
    fetch(`/delete_record/${recordId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('success', `记录 #${recordId} 已删除`);
            
            // 从内存中删除该记录
            allFaceRecords = allFaceRecords.filter(record => record.id != recordId);
            
            // 更新分页
            totalPages = Math.ceil(allFaceRecords.length / recordsPerPage);
            
            // 如果当前页没有数据了，则返回上一页
            if (currentPage > totalPages) {
                currentPage = Math.max(1, totalPages);
            }
            
            updatePaginationInfo();
            renderCurrentPageRecords();
        } else {
            showToast('error', `删除失败: ${data.message || '未知错误'}`);
        }
    })
    .catch(error => {
        console.error('删除记录出错:', error);
        showToast('error', '删除记录时发生错误');
    });
}

// 加载人脸记录数据
function loadFaceRecords() {
    console.log('加载人脸记录数据');
    
    return fetch('/face_records')
        .then(response => response.json())
        .then(records => {
            allFaceRecords = records; // 存储所有记录
            totalPages = Math.ceil(records.length / recordsPerPage);
            
            // 更新分页信息
            updatePaginationInfo();
            
            // 渲染当前页数据
            renderCurrentPageRecords();
        })
        .catch(error => {
            console.error('加载人脸记录出错:', error);
        });
}

// 更新分页信息
function updatePaginationInfo() {
    const paginationInfo = document.getElementById('pagination-info');
    if (!paginationInfo) return;
    
    const totalItems = allFaceRecords.length;
    if (totalItems === 0) {
        paginationInfo.textContent = `没有记录`;
        updatePaginationControls();
        return;
    }
    
    const startItem = (currentPage - 1) * recordsPerPage + 1;
    const endItem = Math.min(currentPage * recordsPerPage, totalItems);
    
    // 更新分页信息文本
    paginationInfo.textContent = `显示 ${startItem}-${endItem} 共 ${totalItems} 条`;
    
    // 更新分页控件
    updatePaginationControls();
}

// 更新分页控件
function updatePaginationControls() {
    const pagination = document.querySelector('.pagination');
    if (!pagination) return;
    
    // 获取上一页和下一页按钮
    const prevButton = document.getElementById('prev-page');
    const nextButton = document.getElementById('next-page');
    if (!prevButton || !nextButton) return;
    
    // 清除除了上一页和下一页按钮以外的所有按钮
    const prevLi = prevButton.parentNode;
    const nextLi = nextButton.parentNode;
    
    // 保存上一页和下一页按钮
    const prevClone = prevLi.cloneNode(true);
    const nextClone = nextLi.cloneNode(true);
    
    // 清空分页容器
    pagination.innerHTML = '';
    
    // 添加上一页按钮
    pagination.appendChild(prevClone);
    
    // 添加新的页码按钮
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);
    
    if (endPage - startPage < 4) {
        startPage = Math.max(1, endPage - 4);
    }
    
    for (let i = startPage; i <= endPage; i++) {
        const li = document.createElement('li');
        li.className = `page-item ${i === currentPage ? 'active' : ''}`;
        
        const a = document.createElement('a');
        a.className = 'page-link';
        a.href = '#';
        a.textContent = i;
        a.onclick = function(e) {
            e.preventDefault();
            goToPage(i);
        };
        
        li.appendChild(a);
        pagination.appendChild(li);
    }
    
    // 添加下一页按钮
    pagination.appendChild(nextClone);
    
    // 重新为上一页和下一页按钮添加事件处理程序
    document.getElementById('prev-page').onclick = function(e) {
        e.preventDefault();
        if (currentPage > 1) {
            goToPage(currentPage - 1);
        }
    };
    
    document.getElementById('next-page').onclick = function(e) {
        e.preventDefault();
        if (currentPage < totalPages) {
            goToPage(currentPage + 1);
        }
    };
    
    // 更新上一页和下一页按钮状态
    document.getElementById('prev-page').parentNode.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    document.getElementById('next-page').parentNode.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
}

// 跳转到指定页
function goToPage(page) {
    if (page < 1 || page > totalPages || page === currentPage) {
        return;
    }
    
    currentPage = page;
    updatePaginationInfo();
    renderCurrentPageRecords();
}

// 渲染当前页数据
function renderCurrentPageRecords() {
    const tableBody = document.getElementById('records-table-body');
    const noRecords = document.getElementById('no-records');
    
    if (!tableBody) return;
    
    if (allFaceRecords && allFaceRecords.length > 0) {
        let tableHTML = '';
        
        // 获取当前页的记录
        const startIndex = (currentPage - 1) * recordsPerPage;
        const endIndex = Math.min(startIndex + recordsPerPage, allFaceRecords.length);
        const currentPageRecords = allFaceRecords.slice(startIndex, endIndex);
        
        currentPageRecords.forEach(record => {
            tableHTML += `
                <tr>
                    <td>${record.id}</td>
                    <td>${record.timestamp}</td>
                    <td>${record.faces_count}</td>
                    <td>
                        <button class="btn btn-sm btn-danger delete-record-btn" data-id="${record.id}">
                            <i class="fas fa-trash-alt"></i>
                        </button>
                    </td>
                </tr>
            `;
        });
        
        tableBody.innerHTML = tableHTML;
        
        if (noRecords) {
            noRecords.classList.add('d-none');
        }
    } else {
        tableBody.innerHTML = '';
        
        if (noRecords) {
            noRecords.classList.remove('d-none');
        }
    }
}

// 更新最近活动列表
function updateRecentActivity(records) {
    const activityContainer = document.getElementById('recent-activity');
    
    if (!activityContainer) {
        console.error('找不到recent-activity容器');
        return;
    }
    
    if (records && records.length > 0) {
        let activitiesHTML = '';
        
        records.forEach(record => {
            const timestamp = record.timestamp;
            const facesCount = record.faces_count;
            
            activitiesHTML += `
                <div class="activity-item">
                    <div class="d-flex align-items-center">
                        <div class="activity-icon bg-success">
                            <i class="fas fa-user-check"></i>
                        </div>
                        <div>
                            <p class="mb-0">检测到 ${facesCount} 个人脸</p>
                            <p class="activity-time mb-0">${timestamp}</p>
                        </div>
                    </div>
                </div>
            `;
        });
        
        activityContainer.innerHTML = activitiesHTML;
    } else {
        activityContainer.innerHTML = `
            <div class="no-records-message">
                <i class="fas fa-inbox"></i>
                <p>暂无检测记录</p>
            </div>
        `;
    }
}

// 加载人脸ID列表
function loadFaces() {
    console.log('加载人脸ID列表');
    
    return fetch('/faces')
        .then(response => response.json())
        .then(faces => {
            const faceContainer = document.getElementById('face-id-container');
            const noFaces = document.getElementById('no-faces');
            
            if (!faceContainer) return;
            
            if (faces && faces.length > 0) {
                let facesHTML = '';
                
                faces.forEach(face => {
                    facesHTML += `
                        <div class="face-id-card">
                            <div class="face-image">
                                <img src="/face_image/${face.id}" alt="人脸 ID: ${face.id}" />
                            </div>
                            <div class="face-id-info">
                                <i class="fas fa-user"></i> ID: ${face.id}
                            </div>
                            <div class="face-id-actions">
                                <i class="fas fa-trash-alt delete-face-btn" data-id="${face.id}"></i>
                            </div>
                        </div>
                    `;
                });
                
                faceContainer.innerHTML = facesHTML;
                
                if (noFaces) {
                    noFaces.classList.add('d-none');
                }
            } else {
                faceContainer.innerHTML = '';
                
                if (noFaces) {
                    noFaces.classList.remove('d-none');
                }
            }
        })
        .catch(error => {
            console.error('加载人脸列表出错:', error);
            throw error;
        });
}

// 显示通知消息
function showToast(type, message) {
    // 先移除所有现有通知
    const existingToasts = document.querySelectorAll('.toast-notification');
    existingToasts.forEach(toast => {
        document.body.removeChild(toast);
    });
    
    // 创建新通知
    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
        <span>${message}</span>
    `;
    document.body.appendChild(toast);
    
    // 显示通知动画
    setTimeout(() => {
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                }
            }, 300);
        }, 2000);
    }, 10);
}

// 更新人员统计表格
function updatePersonStatsTable(stats) {
    const tbody = document.querySelector('#person-stats-table tbody');
    tbody.innerHTML = '';
    
    if (!stats || stats.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center">暂无数据</td></tr>';
        return;
    }
    
    stats.forEach(stat => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${stat.face_id}</td>
            <td>${stat.total_detections}</td>
            <td>${new Date(stat.last_seen).toLocaleString('zh-CN')}</td>
            <td>${formatEmotionDistribution(stat.emotion_distribution)}</td>
            <td>${stat.average_fatigue.toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });
}

// 格式化情绪分布
function formatEmotionDistribution(distribution) {
    if (!distribution) return '暂无数据';
    
    return Object.entries(distribution)
        .map(([emotion, percentage]) => `${emotion}: ${percentage.toFixed(1)}%`)
        .join('<br>');
}

// 绘制情绪趋势图
function drawEmotionTrendChart(emotionData) {
    const ctx = document.getElementById('emotion-trend-chart').getContext('2d');
    
    // 销毁现有图表
    if (window.emotionChart) {
        window.emotionChart.destroy();
    }
    
    // 准备数据
    const labels = emotionData.map(d => new Date(d.timestamp).toLocaleTimeString('zh-CN'));
    const datasets = [];
    
    // 为每种情绪创建数据集
    const emotions = new Set(emotionData.flatMap(d => Object.keys(d.emotions)));
    const colors = [
        'rgba(255, 99, 132, 1)',   // 红色
        'rgba(54, 162, 235, 1)',   // 蓝色
        'rgba(255, 206, 86, 1)',   // 黄色
        'rgba(75, 192, 192, 1)',   // 绿色
        'rgba(153, 102, 255, 1)',  // 紫色
        'rgba(255, 159, 64, 1)',   // 橙色
        'rgba(201, 203, 207, 1)'   // 灰色
    ];
    
    emotions.forEach((emotion, index) => {
        const data = emotionData.map(d => d.emotions[emotion] || 0);
        datasets.push({
            label: emotion,
            data: data,
            borderColor: colors[index % colors.length],
            backgroundColor: colors[index % colors.length].replace('1)', '0.2)'),
            fill: false,
            tension: 0.4
        });
    });
    
    // 创建新图表
    window.emotionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: '情绪占比 (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '时间'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '情绪变化趋势'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// 绘制疲劳趋势图
function drawFatigueChart(fatigueData) {
    const ctx = document.getElementById('fatigue-trend-chart').getContext('2d');
    
    // 销毁现有图表
    if (window.fatigueChart) {
        window.fatigueChart.destroy();
    }
    
    // 准备数据
    const labels = fatigueData.map(d => new Date(d.timestamp).toLocaleTimeString('zh-CN'));
    const data = fatigueData.map(d => d.fatigue_level);
    
    // 创建新图表
    window.fatigueChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: '疲劳度',
                data: data,
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: '疲劳度 (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '时间'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '疲劳度变化趋势'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}