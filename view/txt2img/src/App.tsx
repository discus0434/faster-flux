import CircularProgress from '@mui/material/CircularProgress';
import React, { useCallback, useEffect, useRef, useState } from "react";
import { TextField } from "@mui/material";

function App() {
  const [inputPrompt, setInputPrompt] = useState("");
  const [lastPrompt, setLastPrompt] = useState("");
  const [image, setImage] = useState("images/white.jpg");
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const calculateEditDistance = (a: string, b: string) => {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;

    const matrix = [];

    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }
    for (let i = 0; i <= a.length; i++) {
      matrix[0][i] = i;
    }

    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            Math.min(matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
          );
        }
      }
    }

    return matrix[b.length][a.length];
  };

  const fetchImage = useCallback(async (): Promise<void> => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setIsLoading(true);
    try {
      const response = await fetch('/api/txt2img', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputPrompt }),
        signal,
      });
      const data = await response.json();
      const imageUrl = `data:image/png;base64,${data.imageBase64}`;

      setImage(imageUrl);
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        console.error('Error fetching image:', error);
      }
    } finally {
      setIsLoading(false);
    }
  }, [inputPrompt]);

  const handlePromptChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const newPrompt = event.target.value;
    setInputPrompt(newPrompt);
    const editDistance = calculateEditDistance(lastPrompt, newPrompt);

    if (editDistance >= 4) {
      setLastPrompt(newPrompt);
      fetchImage();
    }
  };

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  return (
    <div
      className="App"
      style={{
        backgroundColor: "#282c34",
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        margin: "0",
        color: "#ffffff",
        padding: "20px",
      }}
    >
      <div
        style={{
          backgroundColor: "#282c34",
          alignItems: "center",
          justifyContent: "center",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {isLoading ? (
          <CircularProgress /> // Show spinner when loading
        ) : null}
        <img
          src={image}
          alt={`Generated image for ${lastPrompt}`}
          style={{
            display: "block",
            margin: "0 auto",
            maxWidth: "100%",
            maxHeight: "70%",
            borderRadius: "10px",
          }}
        />
        <TextField
          variant="outlined"
          value={inputPrompt}
          onChange={handlePromptChange}
          style={{
            marginBottom: "20px",
            marginTop: "20px",
            width: "100%",
            maxWidth: "50rem",
            color: "#ffffff",
            borderColor: "#ffffff",
            borderRadius: "10px",
            backgroundColor: "#ffffff",
          }}
          placeholder="Enter a prompt"
        />
      </div>
    </div>
  );
}

export default App;
