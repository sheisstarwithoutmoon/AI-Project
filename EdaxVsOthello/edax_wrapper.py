import subprocess
import time

class EdaxWrapper:
    def __init__(self, edax_path='./edax.exe', depth=5):
        print("Starting Edax process...")
        args = [edax_path]
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            errors='replace'
        )
        time.sleep(1)
        stderr_output = self.process.stderr.readline()
        if stderr_output:
            print(f"Edax startup error: {stderr_output}")
        if self.process.poll() is not None:
            print("Edax process exited unexpectedly")
            raise RuntimeError("Failed to start Edax")
        self.send_cmd(f'setoption name SearchDepth value {depth}')

    def send_cmd(self, command):
        print(f"Sending command: {command}")
        try:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError) as e:
            print(f"Error sending command: {e}")
            return False

    def get_move(self, fen):
        print(f"Sending FEN: {fen}")
        if not self.send_cmd(f'position fen {fen}'):
            return '0000'
        if not self.send_cmd('go'):
            return '0000'
        start_time = time.time()
        timeout = 10
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            print(f"Edax output: {line}")
            if line.startswith('bestmove'):
                move = line.split()[1].lower()
                print(f"Received bestmove: {move}")
                return move
            if 'plays' in line.lower():
                move = line.split()[-1].lower()
                print(f"Parsed move from 'plays': {move}")
                return move
            if not line or '>' in line:
                continue
            if 'error' in line.lower() or 'invalid' in line.lower():
                print("Edax reported an error")
                return '0000'
        print("Timeout waiting for Edax move")
        return '0000'

    def close(self):
        print("Closing Edax process")
        self.send_cmd('quit')
        try:
            self.process.terminate()
        except Exception as e:
            print(f"Error closing Edax: {e}")