using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading.Tasks;

class SendSingleImage
{
    const string Host = "127.0.0.1";
    const int Port = 6000;

    static async Task Main(string[]? args)
    {
        var path = args?.Length > 0 ? args[0] : @"katze.jpg";
        if (!File.Exists(path))
        {
            Console.WriteLine($"Img not found: {path}");
            return;
        }

        byte[] img = await File.ReadAllBytesAsync(path);
        byte[] len = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(img.Length));

        using var client = new TcpClient();
        await client.ConnectAsync(Host, Port);
        Console.WriteLine($"Connected to {Host}:{Port}");

        using NetworkStream ns = client.GetStream();
        await ns.WriteAsync(len);    // 4-Byte-length
        await ns.WriteAsync(img);    
        Console.WriteLine($"Success: {img.Length} Byte sent.");
    }
}
