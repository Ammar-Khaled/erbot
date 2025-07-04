from flask import Flask, request, jsonify
from rag_chatbot import create_rag_chatbot

app = Flask(__name__)
chatbot = create_rag_chatbot()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'Question is required'}), 400

        response = chatbot.chat(question)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)