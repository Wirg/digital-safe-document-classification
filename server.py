from app import app

if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 2
    app.run(port=8000, debug=True)