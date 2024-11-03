using Amazon;
using Amazon.EC2;
using Amazon.EC2.Model;
using Amazon.Runtime;

namespace DeZero.NET.Processes.CompletionHandler
{
    /// <summary>
    /// Handler to automatically stop the currently running EC2 instance
    /// </summary>
    public class EC2AutoStopHandler : IProcessCompletionHandler
    {
        private readonly string _region;
        private readonly string _accessKey;
        private readonly string _secretKey;
        private readonly bool _terminateInstead;

        public EC2AutoStopHandler(
            string region,
            string accessKey,
            string secretKey,
            bool terminateInstead = false) // trueの場合、stopではなくterminateを実行
        {
            _region = region;
            _accessKey = accessKey;
            _secretKey = secretKey;
            _terminateInstead = terminateInstead;
        }

        public async Task OnProcessComplete(string weightsPath, string recordFilePath)
        {
            try
            {
                Console.WriteLine($"{DateTime.Now} Attempting to stop the current EC2 instance...");

                // 現在のインスタンスIDを取得
                var instanceId = await GetCurrentInstanceId();
                if (string.IsNullOrEmpty(instanceId))
                {
                    Console.WriteLine("Not running on EC2 or unable to get instance ID. Skipping EC2 stop.");
                    return;
                }

                // AWSクライアントの設定
                var ec2Client = new AmazonEC2Client(
                    new BasicAWSCredentials(_accessKey, _secretKey),
                    new AmazonEC2Config { RegionEndpoint = RegionEndpoint.GetBySystemName(_region) }
                );

                if (_terminateInstead)
                {
                    // インスタンスを終了
                    var terminateRequest = new TerminateInstancesRequest
                    {
                        InstanceIds = new List<string> { instanceId }
                    };
                    await ec2Client.TerminateInstancesAsync(terminateRequest);
                    Console.WriteLine($"{DateTime.Now} EC2 instance termination initiated: {instanceId}");
                }
                else
                {
                    // インスタンスを停止
                    var stopRequest = new StopInstancesRequest
                    {
                        InstanceIds = new List<string> { instanceId }
                    };
                    await ec2Client.StopInstancesAsync(stopRequest);
                    Console.WriteLine($"{DateTime.Now} EC2 instance stop initiated: {instanceId}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error stopping/terminating EC2 instance: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        /// <summary>
        /// Retrieves the ID of the currently running EC2 instance
        /// </summary>
        private async Task<string> GetCurrentInstanceId()
        {
            try
            {
                // EC2インスタンスメタデータエンドポイントからインスタンスIDを取得
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(5); // タイムアウトを設定

                // IMDSv2のトークンを取得
                client.DefaultRequestHeaders.Add("X-aws-ec2-metadata-token-ttl-seconds", "21600");
                var tokenResponse = await client.PutAsync(
                    "http://169.254.169.254/latest/api/token",
                    new StringContent(string.Empty)
                );
                var token = await tokenResponse.Content.ReadAsStringAsync();

                // トークンを使用してインスタンスIDを取得
                client.DefaultRequestHeaders.Add("X-aws-ec2-metadata-token", token);
                var response = await client.GetStringAsync(
                    "http://169.254.169.254/latest/meta-data/instance-id"
                );

                return response;
            }
            catch
            {
                // EC2インスタンス上で実行されていない場合やメタデータへのアクセスに失敗した場合
                return null;
            }
        }
    }
}
