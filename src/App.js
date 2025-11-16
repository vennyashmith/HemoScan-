import React, { useState } from 'react';
import CBCForm from './CBCForm';
import ImageUpload from './ImageUpload';
import ResultCard from './ResultCard';

function App() {
  const [cbcData, setCbcData] = useState({
    Gender: 'Male',
    Hemoglobin: '',
    MCV: '',
    MCH: '',
    MCHC: ''
  });
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleCbcChange = (e) => {
    setCbcData({ ...cbcData, [e.target.name]: e.target.value });
  };

  const handleImageChange = (file) => {
    setImage(file);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!image) return alert('Upload an image!');

    const reader = new FileReader();
    reader.onload = async () => {
      const imgBase64 = reader.result;

      // Map Gender to 0/1
      const payload = {
        cbc: {
          Gender: cbcData.Gender === 'Male' ? 0 : 1,
          Hemoglobin: parseFloat(cbcData.Hemoglobin),
          MCV: parseFloat(cbcData.MCV),
          MCH: parseFloat(cbcData.MCH),
          MCHC: parseFloat(cbcData.MCHC)
        },
        image: imgBase64
      };

      try {
        const res = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.prediction) setPrediction(data.prediction);
        else alert(data.error);

      } catch (err) {
        console.error(err);
        alert('Error connecting to backend.');
      }
    };
    reader.readAsDataURL(image);
  };

  return (
    <div className="container my-5">
      <h1 className="text-center text-primary mb-4">HemoScan Prediction</h1>
      <form onSubmit={handleSubmit}>
        <CBCForm formData={cbcData} onChange={handleCbcChange} />
        <ImageUpload handleImageChange={handleImageChange} />
        <button type="submit" className="btn btn-primary w-100 mt-3">Predict</button>
      </form>
      {prediction && <ResultCard result={prediction} />}
    </div>
  );
}

export default App;
