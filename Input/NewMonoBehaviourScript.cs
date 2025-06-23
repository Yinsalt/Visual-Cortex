// TcpFrameSender.cs   –  Attach to any GameObject
using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class TcpFrameSender : MonoBehaviour
{
    [SerializeField] string host = "127.0.0.1";
    [SerializeField] int port = 6000;
    [SerializeField] int jpgQuality = 75;

    TcpClient client;
    NetworkStream stream;
    Texture2D tex;
    byte[] lenBuf = new byte[4];

    void Start()
    {
        Application.runInBackground = true;
        client = new TcpClient();
        client.Connect(host, port);
        stream = client.GetStream();
        Debug.Log($"[TCP] Connected to {host}:{port}");

        Camera cam = GetComponent<Camera>();
        int w = cam.pixelWidth;
        int h = cam.pixelHeight;
        tex = new Texture2D(w, h, TextureFormat.RGB24, false);

        StartCoroutine(SendFrames());
    }

    IEnumerator SendFrames()
    {
        var cam = GetComponent<Camera>();

        while (true)
        {
            yield return new WaitForEndOfFrame();


            tex.ReadPixels(new Rect(0, 0, cam.pixelWidth, cam.pixelHeight), 0, 0);
            tex.Apply(false);                         

            byte[] imgBytes = tex.EncodeToJPG(jpgQuality);

            int n = imgBytes.Length;
            int netLen = IPAddress.HostToNetworkOrder(n);
            lenBuf[0] = (byte)((netLen >> 24) & 0xFF);
            lenBuf[1] = (byte)((netLen >> 16) & 0xFF);
            lenBuf[2] = (byte)((netLen >> 8) & 0xFF);
            lenBuf[3] = (byte)(netLen & 0xFF);

            try
            {
                stream.Write(lenBuf, 0, 4);
                stream.Write(imgBytes, 0, n);
            }
            catch (Exception e)
            {
                Debug.LogError($"[TCP] Transmission Error: {e.Message}");
                break;
            }
        }
    }

    void OnApplicationQuit()
    {
        try { stream?.Close(); client?.Close(); } catch { }
    }
}
