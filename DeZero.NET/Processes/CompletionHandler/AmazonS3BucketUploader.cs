using Amazon;
using Amazon.S3;
using Amazon.S3.Transfer;

namespace DeZero.NET.Processes.CompletionHandler
{
    /// <summary>
    /// Handler to upload training results to an Amazon S3 bucket
    /// </summary>
    /// <param name="s3BucketName">The name of the S3 bucket where the training results will be uploaded.</param>
    /// <param name="s3Region">The AWS region where the S3 bucket is located.</param>
    /// <param name="s3AccessKey">The access key for the AWS account.</param>
    /// <param name="s3SecretKey">The secret key for the AWS account.</param>
    public class AmazonS3BucketUploader(string s3BucketName, string s3Region, string s3AccessKey, string s3SecretKey) : IProcessCompletionHandler
    {
        public string S3BucketName { get; set; } = s3BucketName;
        public string S3Region { get; set; } = s3Region;
        public string S3AccessKey { get; set; } = s3AccessKey;
        public string S3SecretKey { get; set; } = s3SecretKey;

        public async Task OnProcessComplete(string weightsPath, string recordFilePath)
        {
            if (string.IsNullOrEmpty(S3BucketName) || string.IsNullOrEmpty(S3Region) ||
                string.IsNullOrEmpty(S3AccessKey) || string.IsNullOrEmpty(S3SecretKey))
            {
                Console.WriteLine("S3 configuration is not set. Skipping upload.");
                return;
            }

            try
            {
                Console.WriteLine($"{DateTime.Now} Starting S3 upload...");

                var s3Config = new AmazonS3Config
                {
                    RegionEndpoint = RegionEndpoint.GetBySystemName(S3Region)
                };

                using var s3Client = new AmazonS3Client(S3AccessKey, S3SecretKey, s3Config);
                using var transferUtility = new TransferUtility(s3Client);

                // Create timestamped folder name
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var s3FolderPrefix = $"training_results/{timestamp}/";

                // Upload weights folder
                if (Directory.Exists(weightsPath))
                {
                    var weightFiles = Directory.GetFiles(weightsPath, "*", SearchOption.AllDirectories);
                    foreach (var file in weightFiles)
                    {
                        var s3Key = $"{s3FolderPrefix}weights/{Path.GetFileName(file)}";
                        await transferUtility.UploadAsync(file, S3BucketName, s3Key);
                        Console.WriteLine($"Uploaded: {file} -> s3://{S3BucketName}/{s3Key}");
                    }
                }

                // Upload record file
                if (File.Exists(recordFilePath))
                {
                    var s3Key = $"{s3FolderPrefix}{Path.GetFileName(recordFilePath)}";
                    await transferUtility.UploadAsync(recordFilePath, S3BucketName, s3Key);
                    Console.WriteLine($"Uploaded: {recordFilePath} -> s3://{S3BucketName}/{s3Key}");
                }

                Console.WriteLine($"{DateTime.Now} S3 upload completed successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error uploading to S3: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}
