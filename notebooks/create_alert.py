# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a query that checks the percentage of MAE being higher than 7000

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()


alert_query = """
SELECT 
  (case when accuracy_score < 0.6 then 1 else 0 end) as bad_model 
FROM (
  SELECT
    concat(window.start, " - ", window.end) AS Window,
    avg(ROUND(accuracy_score, 2)) as accuracy_score,
    granularity AS Granularity,
    `model_name` AS `Model Id`,
    COALESCE(slice_key, "No slice") AS `Slice key`,
    COALESCE(slice_value, "No slice") AS `Slice value`
  FROM `mlops_dev`.`subhadip`.`model_monitoring_profile_metrics`
  WHERE
    window.start IN (select max(window.start) from `mlops_dev`.`subhadip`.`model_monitoring_profile_metrics`) -- limit to last window
    AND log_type = "INPUT"
    AND ROUND(accuracy_score, 2) is not null
    AND `model_name` = "hotel_reservation_model_basic"
  GROUP BY 
    window.start, window.end, granularity, `model_name`, slice_key, slice_value
)"""


query = w.queries.create(query=sql.CreateQueryRequestQuery(display_name=f'hotel-reservation-alert-query-{time.time_ns()}',
                                                           warehouse_id=srcs[0].warehouse_id,
                                                           description="Alert on hotel reservation model",
                                                           query_text=alert_query))

alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(condition=sql.AlertCondition(operand=sql.AlertConditionOperand(
        column=sql.AlertOperandColumn(name="Accuracy_less_than_0.6"),),
            op=sql.AlertOperator.GREATER_THAN,
            threshold=sql.AlertConditionThreshold(
                value=sql.AlertOperandValue(
                    double_value=45))),
            display_name=f'hotel-reservation-mae-alert-{time.time_ns()}',
            query_id=query.id
        )
    )



# COMMAND ----------

# cleanup
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)