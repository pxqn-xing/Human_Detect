document.addEventListener('DOMContentLoaded', function() {
    console.log('登录页面初始化');
    
    // 打开注册模态框
    const registerLink = document.getElementById('register-link');
    if (!registerLink) {
        console.error('未找到register-link元素');
    } else {
        console.log('找到注册链接元素');
        const registerModalElement = document.getElementById('registerModal');
        if (!registerModalElement) {
            console.error('未找到registerModal元素');
        } else {
            const registerModal = new bootstrap.Modal(registerModalElement);
            
            registerLink.onclick = function(e) {
                console.log('点击了注册链接');
                e.preventDefault();
                registerModal.show();
            };
        }
    }
    
    // 登录表单提交
    const loginForm = document.getElementById('login-form');
    const alertContainer = document.getElementById('alert-container');
    
    if (!loginForm) {
        console.error('未找到login-form元素');
    } else {
        console.log('找到登录表单元素');
        
        loginForm.onsubmit = function(e) {
            console.log('提交登录表单');
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            if (!username || !password) {
                showAlert(alertContainer, '请输入用户名和密码', 'danger');
                return;
            }
            
            console.log('发送登录请求...');
            // 发送登录请求
            fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            })
            .then(response => {
                console.log('登录响应状态:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('登录响应数据:', data);
                if (data.success) {
                    console.log('登录成功，重定向到/dashboard');
                    window.location.href = '/dashboard';
                } else {
                    showAlert(alertContainer, data.message || '登录失败，请检查用户名和密码', 'danger');
                }
            })
            .catch(error => {
                console.error('登录错误:', error);
                showAlert(alertContainer, '连接服务器失败，请稍后再试', 'danger');
            });
        };
    }
    
    // 注册表单提交
    const registerBtn = document.getElementById('register-btn');
    const registerAlertContainer = document.getElementById('register-alert-container');
    
    if (!registerBtn) {
        console.error('未找到register-btn元素');
    } else {
        console.log('找到注册按钮元素');
        
        registerBtn.onclick = function() {
            console.log('点击了注册按钮');
            const username = document.getElementById('register-username').value;
            const password = document.getElementById('register-password').value;
            const confirmPassword = document.getElementById('confirm-password').value;
            
            if (!username || !password || !confirmPassword) {
                showAlert(registerAlertContainer, '请填写所有必填字段', 'danger');
                return;
            }
            
            if (password !== confirmPassword) {
                showAlert(registerAlertContainer, '两次输入的密码不一致', 'danger');
                return;
            }
            
            console.log('发送注册请求...');
            // 发送注册请求
            fetch('/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            })
            .then(response => {
                console.log('注册响应状态:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('注册响应数据:', data);
                if (data.success) {
                    showAlert(registerAlertContainer, '注册成功！请登录', 'success');
                    // 清空表单
                    document.getElementById('register-form').reset();
                    // 3秒后关闭模态框
                    setTimeout(() => {
                        const registerModal = bootstrap.Modal.getInstance(document.getElementById('registerModal'));
                        if (registerModal) {
                            registerModal.hide();
                            // 填充登录表单
                            document.getElementById('username').value = username;
                            document.getElementById('password').value = '';
                        }
                    }, 2000);
                } else {
                    showAlert(registerAlertContainer, data.message || '注册失败，请稍后再试', 'danger');
                }
            })
            .catch(error => {
                console.error('注册错误:', error);
                showAlert(registerAlertContainer, '连接服务器失败，请稍后再试', 'danger');
            });
        };
    }
    
    // 检查登录状态
    console.log('检查登录状态...');
    fetch('/check_login')
        .then(response => response.json())
        .then(data => {
            console.log('登录状态:', data);
            if (data.logged_in) {
                window.location.href = '/dashboard';
            }
        })
        .catch(error => {
            console.error('检查登录状态错误:', error);
        });
});

// 显示提示信息
function showAlert(container, message, type) {
    if (!container) {
        console.error('提示容器不存在');
        return;
    }
    
    console.log(`显示${type}提示:`, message);
    container.innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    // 5秒后自动关闭提示
    setTimeout(() => {
        const alert = container.querySelector('.alert');
        if (alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }
    }, 5000);
} 