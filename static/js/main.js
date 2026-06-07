/* main.js */
window.addEventListener('DOMContentLoaded', () => {
  document.body.classList.add('loaded');
  loadDarkMode();
  initFileInput();
  initDropZone();
  animateBars();
});

function toggleDarkMode() {
  const isDark = document.body.classList.toggle('dark');
  localStorage.setItem('darkMode', isDark ? '1' : '0');
  const btn = document.querySelector('.btn-nav-ghost');
  if (btn) btn.textContent = isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
}

function loadDarkMode() {
  if (localStorage.getItem('darkMode') === '1') {
    document.body.classList.add('dark');
    const btn = document.querySelector('.btn-nav-ghost');
    if (btn) btn.textContent = '☀️ Light Mode';
  }
}

function initFileInput() {
  const input = document.getElementById('fileInput');
  const label = document.getElementById('fileName');
  if (!input || !label) return;
  input.addEventListener('change', () => {
    if (input.files[0]) { label.textContent = input.files[0].name; label.style.color = '#16a34a'; }
    else { label.textContent = 'No file chosen'; label.style.color = ''; }
  });
}

function initDropZone() {
  const zone = document.getElementById('dropZone');
  const input = document.getElementById('fileInput');
  if (!zone || !input) return;

  ['dragenter','dragover'].forEach(e => zone.addEventListener(e, ev => {
    ev.preventDefault(); zone.classList.add('drag-active');
  }));
  ['dragleave','dragend'].forEach(e => zone.addEventListener(e, () => zone.classList.remove('drag-active')));
  zone.addEventListener('drop', ev => {
    ev.preventDefault(); zone.classList.remove('drag-active');
    if (ev.dataTransfer.files[0]) {
      const dt = new DataTransfer(); dt.items.add(ev.dataTransfer.files[0]); input.files = dt.files;
      const label = document.getElementById('fileName');
      if (label) { label.textContent = ev.dataTransfer.files[0].name; label.style.color = '#16a34a'; }
    }
  });
  window.addEventListener('paste', ev => {
    const items = ev.clipboardData && ev.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        const dt = new DataTransfer(); dt.items.add(file); input.files = dt.files;
        const label = document.getElementById('fileName');
        if (label) { label.textContent = 'Pasted image'; label.style.color = '#16a34a'; }
        break;
      }
    }
  });
}

function animateBars() {
  document.querySelectorAll('.progress-fill').forEach(fill => {
    const w = fill.style.width; fill.style.width = '0';
    requestAnimationFrame(() => setTimeout(() => { fill.style.width = w; }, 100));
  });
}