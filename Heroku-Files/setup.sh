mkdir -p ~/.streamlit/
echo "
[general]n
email = "es.sims11@gmail.com"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml