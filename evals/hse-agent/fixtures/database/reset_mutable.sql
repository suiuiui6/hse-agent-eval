DELETE FROM hse_agent_execution_log
WHERE execution_id BETWEEN 980000 AND 989999;

DELETE FROM hse_reminder
WHERE id BETWEEN 980000 AND 989999;

DELETE FROM hse_report_task
WHERE task_id BETWEEN 980000 AND 989999
   OR create_by = 'agent-eval';

DELETE FROM hse_weekly_report
WHERE report_id BETWEEN 980000 AND 989999;

DELETE FROM hse_ai_alert
WHERE alert_id BETWEEN 980000 AND 989999;
