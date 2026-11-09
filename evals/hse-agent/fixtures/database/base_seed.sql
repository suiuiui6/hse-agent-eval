CREATE TABLE IF NOT EXISTS hse_knowledge_index (
    file_id BIGINT NOT NULL COMMENT '源文件ID(hse_file.file_id)',
    document_id VARCHAR(128) DEFAULT NULL COMMENT '外部知识库文档ID(opaque)',
    ingest_job_id VARCHAR(128) DEFAULT NULL COMMENT '外部导入任务ID',
    index_status VARCHAR(32) DEFAULT NULL COMMENT 'pending/indexing/done/failed',
    index_error VARCHAR(500) DEFAULT NULL,
    index_progress INT DEFAULT NULL,
    index_message VARCHAR(500) DEFAULT NULL,
    create_time DATETIME DEFAULT NULL,
    update_time DATETIME DEFAULT NULL,
    PRIMARY KEY (file_id),
    KEY idx_hse_knowledge_index_ingest_job_id (ingest_job_id),
    KEY idx_hse_knowledge_index_document_id (document_id)
) COMMENT='通用知识库源文件索引状态侧表';

ALTER TABLE hse_agent_execution_log AUTO_INCREMENT = 980000;
ALTER TABLE hse_reminder AUTO_INCREMENT = 980000;
ALTER TABLE hse_weekly_report AUTO_INCREMENT = 980000;

INSERT INTO sys_dept (
    dept_id,
    parent_id,
    ancestors,
    dept_name,
    order_num,
    leader,
    phone,
    email,
    status,
    del_flag,
    create_by,
    create_time,
    update_by,
    update_time
) VALUES
    (
        210,
        100,
        '0,100',
        '评测项目一部',
        210,
        '评测负责人一',
        '13800000210',
        'agent-eval-210@example.invalid',
        '0',
        '0',
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW()
    ),
    (
        220,
        100,
        '0,100',
        '评测项目二部',
        220,
        '评测负责人二',
        '13800000220',
        'agent-eval-220@example.invalid',
        '0',
        '0',
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW()
    )
ON DUPLICATE KEY UPDATE
    parent_id = VALUES(parent_id),
    ancestors = VALUES(ancestors),
    dept_name = VALUES(dept_name),
    order_num = VALUES(order_num),
    leader = VALUES(leader),
    phone = VALUES(phone),
    email = VALUES(email),
    status = VALUES(status),
    del_flag = VALUES(del_flag),
    update_by = VALUES(update_by),
    update_time = VALUES(update_time);

INSERT INTO sys_user (
    user_id,
    dept_id,
    user_name,
    nick_name,
    user_type,
    email,
    phonenumber,
    sex,
    avatar,
    password,
    status,
    del_flag,
    login_ip,
    login_date,
    pwd_update_date,
    create_by,
    create_time,
    update_by,
    update_time,
    remark
) VALUES
    (
        980000,
        210,
        'agent_eval_admin',
        '评测管理员',
        '00',
        'agent-eval-admin@example.invalid',
        '13898000000',
        '0',
        '',
        '$2a$10$7JB720yubVSZvUI0rEqK/.VqGOZTH.ulu33dHOiBE8ByOhJIrdAu2',
        '0',
        '0',
        '127.0.0.1',
        NULL,
        NOW(),
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        'HSE Agent 评测管理员'
    ),
    (
        980001,
        210,
        'hse_user',
        '评测项目一部用户',
        '00',
        'hse-user@example.invalid',
        '13898000001',
        '0',
        '',
        '$2a$10$7JB720yubVSZvUI0rEqK/.VqGOZTH.ulu33dHOiBE8ByOhJIrdAu2',
        '0',
        '0',
        '127.0.0.1',
        NULL,
        NOW(),
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        'HSE Agent 评测本部门用户'
    ),
    (
        980002,
        220,
        'other_dept_user',
        '评测项目二部用户',
        '00',
        'other-dept-user@example.invalid',
        '13898000002',
        '0',
        '',
        '$2a$10$7JB720yubVSZvUI0rEqK/.VqGOZTH.ulu33dHOiBE8ByOhJIrdAu2',
        '0',
        '0',
        '127.0.0.1',
        NULL,
        NOW(),
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        'HSE Agent 评测跨部门用户'
    )
ON DUPLICATE KEY UPDATE
    dept_id = VALUES(dept_id),
    user_name = VALUES(user_name),
    nick_name = VALUES(nick_name),
    user_type = VALUES(user_type),
    email = VALUES(email),
    phonenumber = VALUES(phonenumber),
    sex = VALUES(sex),
    avatar = VALUES(avatar),
    password = VALUES(password),
    status = VALUES(status),
    del_flag = VALUES(del_flag),
    pwd_update_date = VALUES(pwd_update_date),
    update_by = VALUES(update_by),
    update_time = VALUES(update_time),
    remark = VALUES(remark);

INSERT INTO sys_role (
    role_id,
    role_name,
    role_key,
    role_sort,
    data_scope,
    menu_check_strictly,
    dept_check_strictly,
    status,
    del_flag,
    create_by,
    create_time,
    update_by,
    update_time,
    remark
) VALUES
    (
        980100,
        '评测管理员',
        'admin',
        980,
        '1',
        1,
        1,
        '0',
        '0',
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        'HSE Agent 评测管理员角色'
    ),
    (
        980101,
        '评测HSE用户',
        'hse_eval_user',
        981,
        '4',
        1,
        1,
        '0',
        '0',
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        'HSE Agent 评测普通用户角色'
    )
ON DUPLICATE KEY UPDATE
    role_name = VALUES(role_name),
    role_key = VALUES(role_key),
    role_sort = VALUES(role_sort),
    data_scope = VALUES(data_scope),
    menu_check_strictly = VALUES(menu_check_strictly),
    dept_check_strictly = VALUES(dept_check_strictly),
    status = VALUES(status),
    del_flag = VALUES(del_flag),
    update_by = VALUES(update_by),
    update_time = VALUES(update_time),
    remark = VALUES(remark);

INSERT INTO sys_menu (
    menu_id,
    menu_name,
    parent_id,
    order_num,
    path,
    component,
    query,
    route_name,
    is_frame,
    is_cache,
    menu_type,
    visible,
    status,
    perms,
    icon,
    create_by,
    create_time,
    update_by,
    update_time,
    remark
) VALUES (
    980200,
    'HSE智能问答评测权限',
    0,
    980,
    '',
    NULL,
    NULL,
    '',
    1,
    0,
    'F',
    '0',
    '0',
    'hse:ai:query',
    '#',
    'agent-eval',
    NOW(),
    'agent-eval',
    NOW(),
    'HSE Agent 评测专用权限'
)
ON DUPLICATE KEY UPDATE
    menu_name = VALUES(menu_name),
    parent_id = VALUES(parent_id),
    order_num = VALUES(order_num),
    menu_type = VALUES(menu_type),
    visible = VALUES(visible),
    status = VALUES(status),
    perms = VALUES(perms),
    update_by = VALUES(update_by),
    update_time = VALUES(update_time),
    remark = VALUES(remark);

INSERT INTO sys_user_role (user_id, role_id) VALUES
    (980000, 980100),
    (980001, 980101),
    (980002, 980101)
ON DUPLICATE KEY UPDATE
    role_id = VALUES(role_id);

INSERT INTO sys_role_menu (role_id, menu_id) VALUES
    (980100, 980200),
    (980101, 980200)
ON DUPLICATE KEY UPDATE
    menu_id = VALUES(menu_id);

INSERT INTO sys_config (
    config_id,
    config_name,
    config_key,
    config_value,
    config_type,
    create_by,
    create_time,
    update_by,
    update_time,
    remark
)
SELECT
    986000,
    '评测环境验证码开关',
    'engine.account.captchaEnabled',
    'false',
    'Y',
    'agent-eval',
    NOW(),
    'agent-eval',
    NOW(),
    '隔离评测数据库禁用登录验证码'
FROM DUAL
WHERE NOT EXISTS (
    SELECT 1
    FROM sys_config
    WHERE config_key = 'engine.account.captchaEnabled'
);

UPDATE sys_config
SET
    config_value = 'false',
    update_by = 'agent-eval',
    update_time = NOW()
WHERE config_key = 'engine.account.captchaEnabled';

INSERT INTO hse_report_task (
    task_id,
    task_code,
    module_type,
    template_folder_id,
    task_name,
    responsible_dept_id,
    period_type,
    period_value,
    start_time,
    deadline_time,
    status,
    del_flag,
    required_file_count,
    submitted_file_count,
    create_by,
    create_time,
    update_by,
    update_time,
    remark
) VALUES
    (
        981000,
        'agent_eval_task_981000',
        'qhse',
        NULL,
        '项目一部月度安全报表',
        210,
        'month',
        '2026-07',
        '2026-07-01 00:00:00',
        '2026-07-20 18:00:00',
        'todo',
        '0',
        1,
        0,
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        '评测待办任务'
    ),
    (
        981001,
        'agent_eval_task_981001',
        'qhse',
        NULL,
        '项目一部逾期隐患整改报表',
        210,
        'month',
        '2026-06',
        '2026-06-01 00:00:00',
        '2026-06-30 18:00:00',
        'overdue',
        '0',
        2,
        0,
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        '评测逾期任务'
    ),
    (
        981002,
        'agent_eval_task_981002',
        'supervision',
        NULL,
        '项目二部安全监督周报',
        220,
        'week',
        '2026-W29',
        '2026-07-13 00:00:00',
        '2026-07-19 18:00:00',
        'todo',
        '0',
        1,
        0,
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        '评测跨部门待办任务'
    ),
    (
        981003,
        'agent_eval_task_981003',
        'branch',
        NULL,
        '项目二部已提交安全检查表',
        220,
        'month',
        '2026-07',
        '2026-07-01 00:00:00',
        '2026-07-15 18:00:00',
        'submitted',
        '0',
        1,
        1,
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        '评测已提交任务'
    ),
    (
        981004,
        'agent_eval_task_981004',
        'professional',
        NULL,
        '项目一部已关闭环保台账',
        210,
        'month',
        '2026-05',
        '2026-05-01 00:00:00',
        '2026-05-31 18:00:00',
        'closed',
        '0',
        1,
        1,
        'agent-eval',
        NOW(),
        'agent-eval',
        NOW(),
        '评测已关闭任务'
    )
ON DUPLICATE KEY UPDATE
    task_code = VALUES(task_code),
    module_type = VALUES(module_type),
    template_folder_id = VALUES(template_folder_id),
    task_name = VALUES(task_name),
    responsible_dept_id = VALUES(responsible_dept_id),
    period_type = VALUES(period_type),
    period_value = VALUES(period_value),
    start_time = VALUES(start_time),
    deadline_time = VALUES(deadline_time),
    status = VALUES(status),
    del_flag = VALUES(del_flag),
    required_file_count = VALUES(required_file_count),
    submitted_file_count = VALUES(submitted_file_count),
    update_by = VALUES(update_by),
    update_time = VALUES(update_time),
    remark = VALUES(remark);

INSERT INTO hse_ai_alert (
    alert_id,
    alert_type,
    severity,
    title,
    description,
    source_type,
    source_id,
    is_read,
    created_at
) VALUES
    (
        982000,
        'overdue',
        'critical',
        '项目一部隐患整改已逾期',
        '任务981001超过截止时间仍未提交',
        'task',
        981001,
        '0',
        '2026-07-17 08:00:00'
    ),
    (
        982001,
        'trend',
        'warning',
        '项目二部安全报表提交率下降',
        '项目二部近两周报表提交率低于评测基线',
        'dept',
        220,
        '0',
        '2026-07-17 08:05:00'
    ),
    (
        982002,
        'compliance',
        'warning',
        '项目一部法规索引待复核',
        '法规索引占位记录需要完成合规复核',
        'knowledge_index',
        983000,
        '1',
        '2026-07-17 08:10:00'
    ),
    (
        982003,
        'gap',
        'info',
        '项目二部本周缺少环保资料',
        '当前评测周期未发现项目二部环保资料上传记录',
        'dept',
        220,
        '0',
        '2026-07-17 08:15:00'
    )
ON DUPLICATE KEY UPDATE
    alert_type = VALUES(alert_type),
    severity = VALUES(severity),
    title = VALUES(title),
    description = VALUES(description),
    source_type = VALUES(source_type),
    source_id = VALUES(source_id),
    is_read = VALUES(is_read),
    created_at = VALUES(created_at);

INSERT INTO hse_knowledge_index (
    file_id,
    document_id,
    ingest_job_id,
    index_status,
    index_error,
    index_progress,
    index_message,
    create_time,
    update_time
) VALUES
    (
        983000,
        'agent-eval-regulation-001',
        'agent-eval-ingest-001',
        'done',
        NULL,
        100,
        '评测法规索引占位记录一',
        NOW(),
        NOW()
    ),
    (
        983001,
        'agent-eval-regulation-002',
        'agent-eval-ingest-002',
        'done',
        NULL,
        100,
        '评测法规索引占位记录二',
        NOW(),
        NOW()
    ),
    (
        983002,
        NULL,
        'agent-eval-ingest-003',
        'pending',
        NULL,
        0,
        '评测法规索引占位记录三',
        NOW(),
        NOW()
    )
ON DUPLICATE KEY UPDATE
    document_id = VALUES(document_id),
    ingest_job_id = VALUES(ingest_job_id),
    index_status = VALUES(index_status),
    index_error = VALUES(index_error),
    index_progress = VALUES(index_progress),
    index_message = VALUES(index_message),
    update_time = VALUES(update_time);
