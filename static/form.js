document.querySelector('#chat-form').onsubmit = function () {
    // Get values
    const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    const user_input = document.querySelector('#chat-input').value;
    document.querySelector('#chat-input').value = '';
    const chat = document.querySelector('#chat');
  
    // set user message
    const user_msg = document.createElement('article');
  
    const user_msg_header = document.createElement('header');
    user_msg_header.innerHTML = '<kbd>User</kbd>';
  
    const user_msg_content = document.createElement('p');
    user_msg_content.innerHTML = user_input;
  
    user_msg.append(user_msg_header);
    user_msg.append(user_msg_content);
  
    chat.append(user_msg);
  
    // Send input
    fetch('/send/', {
      method: 'POST',
      headers: { 'X-CSRFToken': csrftoken },
      body: JSON.stringify({
        user_input: user_input,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        const response_msg = document.createElement('article');
        const response_msg_header = document.createElement('header');
        response_msg_header.innerHTML = '<kbd>System</kbd>';
  
        const response_msg_content = document.createElement('p');
        response_msg_content.innerHTML = data['content'];
  
        response_msg.append(response_msg_header);
        response_msg.append(response_msg_content);
  
        chat.append(response_msg);
      });
  
    return false;
  };

