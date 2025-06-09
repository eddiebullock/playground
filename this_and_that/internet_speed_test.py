import speedtest as st

def Speed_Test():
    test = st.Speedtest()

    down_speed = test.download()
    down_speed = round(down_speed / 1000000, 2)
    print("Download Speed: ", down_speed, "Mbps")

    up_speed = test.upload()
    up_speed = round(up_speed / 1000000, 2)
    print("Upload speed: ", up_speed, "Mbps")

    ping_result = test.results.ping
    print("Ping: ", ping_result, "ms")

if __name__ == "__main__":
    Speed_Test() 