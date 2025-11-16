// ResultCard.js
import React from 'react';

const ResultCard = ({ result }) => {
  return (
    <div className="container mt-3">
      <div className="card shadow-sm">
        <div className="card-body">
          <h5 className="card-title text-warning">Prediction Result</h5>
          <p className="card-text fs-5">{result}</p>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;
