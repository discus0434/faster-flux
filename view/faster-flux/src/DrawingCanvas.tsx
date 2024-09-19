import React, { useRef, useEffect } from 'react';

interface DrawingCanvasProps {
  onCanvasChange: (dataUrl: string) => void;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({ onCanvasChange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    }
  }, []);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDrawing.current = true;
    draw(e);
  };

  const finishDrawing = () => {
    isDrawing.current = false;
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.beginPath(); // パスをリセット

        // オフスクリーンキャンバスを作成
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = canvas.width;
        offscreenCanvas.height = canvas.height;
        const offscreenCtx = offscreenCanvas.getContext('2d');
        if (offscreenCtx) {
          // 背景を白で塗りつぶす
          offscreenCtx.fillStyle = 'white';
          offscreenCtx.fillRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);

          // 元のキャンバスをオフスクリーンキャンバスに描画
          offscreenCtx.drawImage(canvas, 0, 0);

          // オフスクリーンキャンバスからデータURLを取得（JPEG形式でエクスポート）
          const dataUrl = offscreenCanvas.toDataURL('image/jpeg', 1.0);
          onCanvasChange(dataUrl);
        } else {
          // オフスクリーンコンテキストが取得できない場合のフォールバック
          const dataUrl = canvas.toDataURL('image/jpeg', 1.0);
          onCanvasChange(dataUrl);
        }
      }
    }
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (ctx && canvas) {
      const rect = canvas.getBoundingClientRect();
      ctx.lineWidth = 100;
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'gray';

      ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      style={{ border: '1px solid #000', width: '100%', height: '100%' }}
      onMouseDown={startDrawing}
      onMouseUp={finishDrawing}
      onMouseMove={draw}
      onMouseLeave={finishDrawing}
    />
  );
};

export default DrawingCanvas;
