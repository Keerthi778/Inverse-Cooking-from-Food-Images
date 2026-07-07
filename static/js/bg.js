/* Animated neural network background */
(function () {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, nodes = [], frame = 0;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkNode() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      r: 1.5 + Math.random() * 2,
      pulse: Math.random() * Math.PI * 2
    };
  }

  function init() {
    resize();
    nodes = Array.from({ length: 80 }, mkNode);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    frame++;

    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy; n.pulse += 0.02;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < 130) {
          const alpha = (1 - d / 130) * 0.18;
          ctx.beginPath();
          ctx.strokeStyle = `rgba(232,93,36,${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }

    nodes.forEach(n => {
      const glow = 0.4 + 0.3 * Math.sin(n.pulse);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r * (1 + 0.3 * Math.sin(n.pulse)), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(232,93,36,${glow})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  init();
  draw();

  /* Food emoji particles */
  const food = ['🍕','🍣','🥗','🍜','🥘','🍛','🥞','🍱','🥙','🍝'];
  const wrap = document.getElementById('particles');
  if (wrap) {
    for (let i = 0; i < 10; i++) {
      const el = document.createElement('div');
      el.textContent = food[i % food.length];
      const size = 18 + Math.random() * 20;
      el.style.cssText = `
        position:absolute;
        font-size:${size}px;
        left:${Math.random()*100}%;
        top:${Math.random()*100}%;
        animation: floatParticle ${6+Math.random()*8}s ease-in-out infinite;
        animation-delay: ${Math.random()*6}s;
        opacity: 0.07;
        pointer-events:none;
        user-select:none;
      `;
      wrap.appendChild(el);
    }
  }
})();