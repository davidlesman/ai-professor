# Project 5: AI Rate My Professor - Headstarter Fellowship
## Overview

This project is part of the Headstarter Fellowship, focusing on the development of an AI Rate My Professor using Gemini. The project was built using Django for the backend, JavaScript for the frontend interactions, and PicoCSS for styling. The AI agent is designed to assist users with various support tasks, leveraging AI capabilities provided by Gemini.
Features

* AI-Powered Support: The AI agent is capable of handling user queries and guiding users through different processes.
* User-Friendly Interface: The frontend is built with JavaScript and styled with PicoCSS to ensure a responsive and intuitive user experience.
* Django Backend: The backend is powered by Django, offering a robust framework for managing the AI agent's data and interactions.
* Gemini Integration: The AI functionalities are integrated using Gemini, allowing for advanced natural language processing and support capabilities. The embeddings feature allows for quick information retrieval about the professors.

## Technologies Used

* Django: Used for the backend framework, handling database interactions, and serving the web application.
* JavaScript: Provides interactivity on the frontend, enabling dynamic content and real-time user interactions.
* PicoCSS: A lightweight CSS framework used for styling, ensuring a minimalist and responsive design.
* Gemini: The AI engine that powers the support agent, providing AI-driven responses and assistance.

## Installation

To get started with the project, follow these steps:

Clone the repository:

```bash
git clone https://github.com/davidlesman/ai-professor.git
cd ai-professor
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Apply migrations:

```bash
python manage.py migrate
```

Run the development server:

```bash
python manage.py runserver
```

Access the application:
Open your web browser and navigate to http://localhost:8000.


## Usage

* Navigate to the Support Agent: Once the server is running, you can interact with the AI support agent on the main page.
* Ask Questions: Type your queries into the chat interface, and the AI agent will respond with relevant information or guidance.
* Explore Features: The agent is equipped with several support functionalities, such as troubleshooting, providing information, and more.

## Demo
[YouTube](https://youtu.be/9J_gFi-O-9k)

## Acknowledgments

* Headstarter Fellowship: For the opportunity and guidance throughout the project.
* Gemini AI: For providing the AI capabilities that power the support agent.
* Django & PicoCSS: For the powerful tools that made this project possible.
