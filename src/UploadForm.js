import React, { useState } from "react";

function UploadForm() {
  const [image, setImage] = useState(null);
  const [gender, setGender] = useState("male");
  const [hemoglobin, setHemoglobin] = useState("");
  const [mch, setMCH] = useState("");
  const [mchc, setMCHC] = useState("");
  const [mcv, setMCV] = useState("");
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) return alert("Upload an image!");

    // Convert image to base64
    const reader = new FileReader();
    reader.onload = async () => {
      const imgBase64 = reader.result;

      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imgBase64,
          gender,
          hemoglobin: parseFloat(hemoglobin),
          mch: parseFloat(mch),
          mchc: parseFloat(mchc),
          mcv: parseFloat(mcv)
        }),
      });

      const data = await res.json();
      if (data.prediction !== undefined) setPrediction(data.prediction);
      else alert(data.error);
    };
    reader.readAsDataURL(image);
  };

  return (
    <div>
      <h2>Anemia Detection</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={e => setImage(e.target.files[0])} required />
        <select value={gender} onChange={e => setGender(e.target.value)}>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>
        <input placeholder="Hemoglobin" value={hemoglobin} onChange={e => setHemoglobin(e.target.value)} />
        <input placeholder="MCH" value={mch} onChange={e => setMCH(e.target.value)} />
        <input placeholder="MCHC" value={mchc} onChange={e => setMCHC(e.target.value)} />
        <input placeholder="MCV" value={mcv} onChange={e => setMCV(e.target.value)} />
        <button type="submit">Predict</button>
      </form>
      {prediction !== null && <h3>Prediction: {prediction === 1 ? "Anemic" : "Non-Anemic"}</h3>}
    </div>
  );
}

export default UploadForm;
