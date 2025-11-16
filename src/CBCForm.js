import React from 'react';

const CBCForm = ({ formData, onChange }) => {
  return (
    <div className="container mt-4">
      <h3 className="mb-4 text-primary">CBC Form</h3>
      <div className="mb-3">
        <label className="form-label">Gender</label>
        <select
          className="form-select"
          name="Gender"
          value={formData.Gender}
          onChange={onChange}
          required
        >
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div>
      <div className="mb-3">
        <label className="form-label">Hemoglobin</label>
        <input
          type="number"
          className="form-control"
          name="Hemoglobin"
          value={formData.Hemoglobin}
          onChange={onChange}
          required
        />
      </div>
      <div className="mb-3">
        <label className="form-label">MCV</label>
        <input
          type="number"
          className="form-control"
          name="MCV"
          value={formData.MCV}
          onChange={onChange}
          required
        />
      </div>
      <div className="mb-3">
        <label className="form-label">MCH</label>
        <input
          type="number"
          className="form-control"
          name="MCH"
          value={formData.MCH}
          onChange={onChange}
          required
        />
      </div>
      <div className="mb-3">
        <label className="form-label">MCHC</label>
        <input
          type="number"
          className="form-control"
          name="MCHC"
          value={formData.MCHC}
          onChange={onChange}
          required
        />
      </div>
    </div>
  );
};

export default CBCForm;
