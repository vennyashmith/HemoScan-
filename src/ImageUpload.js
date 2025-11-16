import React from 'react';

const ImageUpload = ({ handleImageChange }) => {
  const onChange = (e) => {
    if (e.target.files.length > 0) {
      handleImageChange(e.target.files[0]);
    }
  };

  return (
    <div className="container mt-4">
      <h3 className="mb-3 text-success">Upload Image</h3>
      <input type="file" accept="image/*" onChange={onChange} className="form-control" />
    </div>
  );
};

export default ImageUpload;
