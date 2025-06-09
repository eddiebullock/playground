import speedtest as st
import time

def Speed_Test():
    try:
        print("Initializing speed test...")
        test = st.Speedtest(secure=True)  # Using secure connection
        
        print("Finding best server...")
        test.get_best_server()
        
        print("Testing download speed...")
        down_speed = test.download()
        down_speed = round(down_speed / 1000000, 2)
        print("Download Speed: ", down_speed, "Mbps")

        print("Testing upload speed...")
        up_speed = test.upload()
        up_speed = round(up_speed / 1000000, 2)
        print("Upload speed: ", up_speed, "Mbps")

        ping_result = test.results.ping
        print("Ping: ", ping_result, "ms")
        
    except st.ConfigRetrievalError as e:
        print("Error: Could not connect to speedtest.net servers.")
        print("This might be due to network restrictions or server blocking.")
        print("Try again later or check your network connection.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please try again later.")

if __name__ == "__main__":
    Speed_Test() 