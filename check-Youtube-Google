import requests
from datetime import datetime
import time
import csv
from requests.exceptions import RequestException

# 🔐 YouTube Data API Configuration
API_KEY = "secret"
CHANNEL_HANDLE = "Controvergent"


def get_channel_id(username):
    """Fetch YouTube channel ID using handle"""
    url = f"https://www.googleapis.com/youtube/v3/channels?part=id&forHandle=@{username}&key={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data["items"][0]["id"] if data.get("items") else None
    except (RequestException, KeyError) as e:
        print(f"Error fetching channel ID: {e}")
        return None


def get_upload_playlist_id(channel_id):
    """Get channel's upload playlist ID"""
    url = f"https://www.googleapis.com/youtube/v3/channels?part=contentDetails&id={channel_id}&key={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except (RequestException, KeyError) as e:
        print(f"Error fetching playlist ID: {e}")
        raise


def get_video_ids(playlist_id):
    """Retrieve all video IDs from upload playlist"""
    video_ids = []
    next_page_token = None

    while True:
        url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&playlistId={playlist_id}&key={API_KEY}"
        if next_page_token:
            url += f"&pageToken={next_page_token}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            video_ids.extend(item["snippet"]["resourceId"]["videoId"] for item in data.get("items", []))
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
        except (RequestException, KeyError) as e:
            print(f"Error fetching videos: {e}")
            break

    return video_ids


def check_video_visibility(video_id):
    """Check video visibility using YouTube API"""
    url = f"https://www.googleapis.com/youtube/v3/videos?part=status&id={video_id}&key={API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        items = response.json().get("items", [])

        if not items:
            return "NOT_FOUND"

        status = items[0]["status"]
        return "PUBLIC" if status.get("privacyStatus") == "public" else status["privacyStatus"]

    except RequestException as e:
        print(f"API Error for {video_id}: {e}")
        return "API_ERROR"


def save_results(video_ids, txt_file="results.txt"):
    """Save results with API quota management"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_file = f"youtube_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(txt_file, "a") as txt_output, open(csv_file, "w", newline='', encoding='utf-8') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(["Timestamp", "Video ID", "Visibility", "Embeddable", "License"])

        for idx, video_id in enumerate(video_ids):
            try:
                # Get detailed video status
                status_url = f"https://www.googleapis.com/youtube/v3/videos?part=status&id={video_id}&key={API_KEY}"
                status_response = requests.get(status_url).json()
                status = status_response.get("items", [{}])[0].get("status", {})

                # Write results
                visibility = status.get("privacyStatus", "UNKNOWN")
                embeddable = status.get("embeddable", False)
                license = status.get("license", "UNKNOWN")

                log_line = f"{timestamp} | {video_id} | {visibility} | {embeddable} | {license}"
                print(log_line)

                txt_output.write(log_line + "\n")
                csv_writer.writerow([timestamp, video_id, visibility, embeddable, license])

                # Manage API quota (10000 units/day)
                time.sleep(1)  # Ensure 1s between requests

            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                continue


def main_process():
    """Main execution with quota monitoring"""
    channel_id = get_channel_id(CHANNEL_HANDLE)
    if not channel_id:
        print("Channel not found. Verify the channel handle and API key.")
        return

    try:
        playlist_id = get_upload_playlist_id(channel_id)
        video_ids = get_video_ids(playlist_id)
        print(f"Found {len(video_ids)} videos. Checking visibility...")
        save_results(video_ids)
    except Exception as e:
        print(f"Process failed: {e}")


if __name__ == "__main__":
    main_process()
