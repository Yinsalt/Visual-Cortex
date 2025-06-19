
import socket
import struct
import threading
import queue
import numpy as np
import cv2

HOST = "0.0.0.0"
PORT = 6000
MAX_QUEUE = 300

frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE)

def recv_all(sock: socket.socket, num_bytes: int) -> bytes:
    data = bytearray()
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            raise ConnectionError("Client disconnected")
        data.extend(packet)
    return bytes(data)

def handle_client(conn: socket.socket, addr):
    print(f"[+] Client {addr} verbunden")
    try:
        while True:
            header = recv_all(conn, 4)
            frame_len = struct.unpack("!I", header)[0]  
            if frame_len == 0 or frame_len > 50_000_000:
                print(f"[!] Ungültige Frame-Länge: {frame_len}")
                break

            jpg_bytes = recv_all(conn, frame_len)

            buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                print("[!] Dekodierung fehlgeschlagen, Frame übersprungen")
                continue

            try:
                frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                print("[!] Frame-Queue voll – älteste Bilder werden verworfen")
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)

    except (ConnectionError, OSError) as e:
        print(f"[!] Verbindung zu {addr} beendet ({e})")
    finally:
        conn.close()
        print(f"[-] Client {addr} getrennt")

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"[•] Lausche auf {HOST}:{PORT}")
        while True:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    threading.Thread(target=start_server, daemon=True).start()

    print("[•] Warte auf das erste Bild …")
    first_frame = frame_queue.get() 
    print("[✓] Erstes Bild erhalten, wird angezeigt")

    cv2.imshow("Erstes empfangenes Bild", first_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
