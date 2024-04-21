import http.server
import http.client
import ssl

"""
## Generate self signed keys
1) openssl genrsa -out ssl.key 2048
2) openssl req -new -key ssl.key -out ssl.csr
3) openssl x509 -req -in ssl.csr -signkey ssl.key -out ssl.crt -days 365

### verify
openssl x509 -in ssl.crt -text -noout
openssl rsa -in ssl.key -check


## use with LMSTUDIO
Create port forward for 8443 (or the port you're using)

### CURL example
curl --insecure  https://<your external IP>:8443/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "LoneStriker/Meta-Llama-3-70B-Instruct-GGUF",
    "messages": [
      { "role": "system", "content": "Always answer in rhymes." },
      { "role": "user", "content": "Introduce yourself." }
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": false
}'
"""

class ProxyHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Forward the request to the local HTTP server
            client = http.client.HTTPConnection("localhost", 1234)
            client.request(
                method=self.command,
                url=self.path,
                body=self.rfile.read(int(self.headers['Content-Length'])),
                headers=self.headers
            )
            response = client.getresponse()

            # Send the response back to the client
            self.send_response(response.status)
            for header, value in response.getheaders():
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(response.read())
        except Exception as e:
            self.send_error(500, str(e))

if __name__ == "__main__":
    server_address = ('', 8443)
    httpd = http.server.HTTPServer(server_address, ProxyHTTPRequestHandler)
    
    # Load SSL certificate and key
    httpd.socket = ssl.wrap_socket(httpd.socket, keyfile="ssl.key", certfile="ssl.crt", server_side=True)
    
    print("Proxy server running on port 8443...")
    httpd.serve_forever()

