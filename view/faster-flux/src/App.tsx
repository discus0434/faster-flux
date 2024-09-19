import React, { useState, useEffect } from "react";
import axios from "axios";
import DrawingCanvas from "./DrawingCanvas";

const App: React.FC = () => {
  const [canvasData, setCanvasData] = useState<string>("");
  const [textInput, setTextInput] = useState<string>("");
  const [outputImage, setOutputImage] = useState<string>("");

  useEffect(() => {
    if (canvasData || textInput) {
      const sendData = setTimeout(() => {
        postData();
      }, 500); // Debounce to prevent too many requests
      return () => clearTimeout(sendData);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvasData, textInput]);

  const handleCanvasChange = (dataUrl: string) => {
    setCanvasData(dataUrl);
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTextInput(e.target.value);
  };

  const postData = async () => {
    try {
      const response = await axios.post(
        `http://127.0.0.1:${process.env.REACT_APP_PORT_NUMBER}/predict`,
        {
          image: canvasData,
          text: textInput,
        }
      );
      setOutputImage(response.data.imageBase64);
    } catch (error) {
      console.error("Error posting data:", error);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <div style={{ display: "flex" }}>
        {/* Left Canvas */}
        <div style={{ width: "45%", aspectRatio: "1 / 1" }}>
          <DrawingCanvas onCanvasChange={handleCanvasChange} />
        </div>
        {/* Right Canvas */}
        <div style={{ width: "45%", aspectRatio: "1 / 1", marginLeft: "10%" }}>
          <canvas
            style={{ border: "1px solid #000", width: "100%", height: "100%" }}
            ref={(canvas) => {
              if (canvas && outputImage) {
                const ctx = canvas.getContext("2d");
                const img = new Image();
                img.onload = () => {
                  if (ctx) {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                  }
                };
                img.src = `data:image/png;base64,${outputImage}`;
              }
            }}
          ></canvas>
        </div>
      </div>
      {/* Text Input */}
      <div style={{ marginTop: "20px" }}>
        <input
          type="text"
          value={textInput}
          onChange={handleTextChange}
          placeholder="Enter text here"
          style={{ width: "100%", padding: "10px", fontSize: "16px" }}
        />
      </div>
    </div>
  );
};

export default App;
