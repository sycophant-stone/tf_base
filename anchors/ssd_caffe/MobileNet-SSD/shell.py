import subprocess
import time
import os
import sys
from pygments import highlight, lexers, formatters



def run_system_commands(cmd_lines, cwd=''):
    for cmd_line in cmd_lines:
        run_system_command(cmd_line, cwd)


def chain_system_commands(cmd_lines, cwd=''):
    cmd_line = '&& \\'.join(cmd_lines)
    run_system_command(cmd_line, cwd)


def run_system_command(cmd_line, cwd='', ignore_err=False):
    hdfs_put_retry = 2
    while True:
        #print('%s %s', cwd, cmd_line)
        cur_cwd = os.getcwd()
        if cwd != '':
            os.chdir(cwd)
        ps = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)

        stdout, stderr = ps.communicate()
        status = ps.wait()
        if len(stdout) > 0:
            print(stdout)

        if cwd != '':
            os.chdir(cur_cwd)

        if status != 0:
            if cmd_line.find('hdfs dfs -put') >= 0 and hdfs_put_retry > 0:
                hdfs_put_retry -= 1
                time.sleep(60)
                continue
            if ignore_err:
                print('%s %s %s' % (cmd_line, stdout, stderr))
            else:
                print('%s %s %s' % (cmd_line, stdout, stderr))
                raise Exception('Exception running command: \n%s\n%s\n%s' % (cmd_line, stdout, stderr))
            return False
        return True


if __name__ == "__main__":
    run_system_command('ls')
