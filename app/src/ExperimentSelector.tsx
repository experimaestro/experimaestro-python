import React from "react";
import Form from "react-bootstrap/Form";
import { useAppSelector } from "./store";
import { actions } from "./reducers";
import { useDispatch } from "react-redux";

export const ExperimentSelector: React.FC = () => {
  const dispatch = useDispatch();
  const experiments = useAppSelector((state) => state.db.experiments);
  const currentExperiment = useAppSelector((state) => state.db.currentExperiment);

  // Hide selector if there are 0 or 1 experiments
  if (experiments.ids.length <= 1) {
    return null;
  }

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value;
    dispatch(actions.selectExperiment(value === "" ? null : value));
  };

  // Helper to get experiment status color
  const getExperimentStatus = (expId: string): "success" | "warning" | "danger" => {
    const exp = experiments.byId[expId];
    if (exp.failed_jobs > 0) return "danger";
    if (exp.finished_jobs < exp.total_jobs) return "warning";
    return "success";
  };

  return (
    <Form.Select
      value={currentExperiment || ""}
      onChange={handleChange}
      style={{ width: "auto", marginLeft: "1rem" }}
      size="sm"
    >
      <option value="">All Experiments</option>
      {experiments.ids.map((expId) => {
        const exp = experiments.byId[expId];
        const status = getExperimentStatus(expId);
        const icon = status === "danger" ? "❌" : status === "warning" ? "⏳" : "✓";
        return (
          <option key={expId} value={expId}>
            {icon} {exp.experiment_id} ({exp.finished_jobs}/{exp.total_jobs} jobs)
          </option>
        );
      })}
    </Form.Select>
  );
};
