import React, { useCallback, useEffect, useRef, useState } from "react";
import { TextField } from "@mui/material";

function App() {
  const [inputPrompt, setInputPrompt] = useState("");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isFetchingRef = useRef(false);
  const pendingPromptRef = useRef<string>("");

  // fetchImage関数を修正し、フェッチ中にプロンプトが変更された場合の処理を追加
  const fetchImage = useCallback(async (prompt: string): Promise<void> => {
    isFetchingRef.current = true;

    try {
      const response = await fetch('/api/txt2img', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: prompt }),
      });
      const data = await response.json();
      const imageBase64 = data.imageBase64;

      // 画像をCanvasに描画
      if (canvasRef.current && imageBase64) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const img = new Image();
          img.onload = () => {
            // キャンバスのサイズを画像サイズに合わせる
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
          };
          img.src = `data:image/png;base64,${imageBase64}`;
        }
      }
    } catch (error) {
      if (error instanceof Error) {
        console.error('画像の取得中にエラーが発生しました:', error);
      }
    } finally {
      isFetchingRef.current = false;
      // フェッチ中にプロンプトが変更されていれば、新たなリクエストを送信
      if (pendingPromptRef.current && pendingPromptRef.current !== prompt) {
        const nextPrompt = pendingPromptRef.current;
        pendingPromptRef.current = "";
        fetchImage(nextPrompt);
      }
    }
  }, []);

  // プロンプトの変更時に即座にfetchImageを呼び出す（フェッチ中の場合はpendingPromptRefに保存）
  useEffect(() => {
    if (inputPrompt.trim() === '') {
      return;
    }

    if (!isFetchingRef.current) {
      fetchImage(inputPrompt);
    } else {
      // フェッチ中であれば、最新のプロンプトをpendingPromptRefに保存
      pendingPromptRef.current = inputPrompt;
    }
  }, [inputPrompt, fetchImage]);

  // プロンプトの変更ハンドラー
  const handlePromptChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    setInputPrompt(event.target.value);
  };

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
        <canvas
          ref={canvasRef}
          style={{
            display: "block",
            width: "100%",
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
          placeholder="プロンプトを入力してください"
        />
      </div>
    </div>
  );
}

export default App;
