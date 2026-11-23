'use strict';
ABSASidebar.create({embedded: true, mount: document.querySelector('#mount')})
  .catch(() => { document.querySelector('#error').textContent = 'Không khởi tạo được extension. Hãy Load unpacked đúng thư mục extension/ và mở lại.'; });
