import http.server
import socketserver
from jinja2 import Environment, FileSystemLoader
import markdown

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('index.md')


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            myvar  = 'place="far-far-away"'
            vars = dict(myvar=myvar)
            md_content = template.render(**vars)
            html_content = markdown.markdown(md_content)
            self.wfile.write(html_content.encode())
        else:
            super().do_GET()


if __name__ == "__main__":
    PORT = 8000
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
