using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Threading.Tasks;
using OpenCvSharp;

class SendVideoFrames
{
    const string Host = "127.0.0.1";
    const int Port = 6000;
    const double TargetFps = 60.0;

    static async Task Main(string[]? args)
    {
        var videoPath = args?.Length > 0 ? args[0] : @"C:*********\Testvid2.mp4";
        if (!System.IO.File.Exists(videoPath))
        {
            Console.WriteLine($"Video not found: {videoPath}");
            return;
        }

        using var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened())
        {
            Console.WriteLine("Could not open video.");
            return;
        }

        using var client = new TcpClient();
        await client.ConnectAsync(Host, Port);
        await using NetworkStream ns = client.GetStream();
        Console.WriteLine($"Connected to {Host}:{Port}");

        var frame = new Mat();
        var interval = TimeSpan.FromSeconds(1.0 / TargetFps);
        var sw = new Stopwatch();

        while (capture.Read(frame))
        {
            if (frame.Empty())
                break;

            Cv2.ImEncode(".jpg", frame, out byte[] jpgBytes);
            byte[] lenPrefix = BitConverter.GetBytes(
                                   IPAddress.HostToNetworkOrder(jpgBytes.Length));
            await ns.WriteAsync(lenPrefix);
            await ns.WriteAsync(jpgBytes);

            // --- 60-fps-
            if (!sw.IsRunning) sw.Start();
            else
            {
                var elapsed = sw.Elapsed;
                if (elapsed < interval)
                    await Task.Delay(interval - elapsed);
                sw.Restart();
            }
        }

        Console.WriteLine("Finished.");
    }
}
