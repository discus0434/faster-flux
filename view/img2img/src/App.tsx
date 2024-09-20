import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import DrawingCanvas from './DrawingCanvas';

const App: React.FC = () => {
  const [canvasData, setCanvasData] = useState<string>('');
  const [textInput, setTextInput] = useState<string>('');
  const [outputImage, setOutputImage] = useState<string>('');

  // キャンバスの参照を保持
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasData || textInput) {
      const sendData = setTimeout(() => {
        postData();
      }, 500); // デバウンス処理
      return () => clearTimeout(sendData);
    }
    // 注意: 必要に応じて依存関係を調整してください
  }, [canvasData, textInput]);

  const handleCanvasChange = (dataUrl: string) => {
    setCanvasData(dataUrl);
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTextInput(e.target.value);
  };

  const postData = async () => {
    try {
      const response = await axios.post('/api/img2img', {
        image: canvasData,
        text: textInput,
      });
      setOutputImage(response.data.imageBase64);
    } catch (error) {
      console.error('データの送信中にエラーが発生しました:', error);
    }
  };

  // outputImageが更新されたときにキャンバスに画像を描画
  useEffect(() => {
    if (outputImage) {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const img = new Image();
          img.onload = () => {
            // キャンバスをクリア
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // キャンバスのサイズを画像サイズに合わせる
            canvas.width = img.width;
            canvas.height = img.height;
            // 画像を描画
            ctx.drawImage(img, 0, 0);
          };
          img.onerror = (e) => {
            console.error('画像の読み込みに失敗しました:', e);
          };
          // `outputImage`がデータURLで始まるかを確認
          if (outputImage.startsWith('data:image')) {
            img.src = outputImage;
          } else {
            img.src = `data:image/webp;base64,${outputImage}`;
          }
        } else {
          console.error('2Dコンテキストが取得できませんでした。');
        }
      } else {
        console.error('キャンバス要素が見つかりませんでした。');
      }
    }
  }, [outputImage]);

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ display: 'flex' }}>
        {/* 左側のキャンバス */}
        <div style={{ width: '45%', aspectRatio: '1 / 1' }}>
          <DrawingCanvas onCanvasChange={handleCanvasChange} />
        </div>
        {/* 右側のキャンバス */}
        <div style={{ width: '45%', aspectRatio: '1 / 1', marginLeft: '10%' }}>
          <canvas
            ref={canvasRef}
            style={{ border: '1px solid #000', width: '100%', height: '100%' }}
          />
        </div>
      </div>
      {/* テキスト入力 */}
      <div style={{ marginTop: '20px' }}>
        <input
          type="text"
          value={textInput}
          onChange={handleTextChange}
          placeholder="文字を入力してください"
          style={{ width: '100%', padding: '10px', fontSize: '16px' }}
        />
      </div>
    </div>
  );
};

export default App;
