#!/bin/bash
# Launch one on-demand g4dn.xlarge (NVIDIA T4 16 GB, the free-Colab card) to execute notebooks/colab_free_tier.ipynb.
# Adapted from hev/rerank-bench deploy/launch.sh: price from the AWS Price List API, a budget gate on the worst case,
# and a hard stop. The instance shuts itself down after MAX_MINUTES and shutdown means terminate, so a lost session
# can't leave it running. Usage: scripts/t4/launch.sh SPENT_SO_FAR_USD [--dry-run]
# Env: AWS_PROFILE (default mesh-916), MAX_MINUTES (default 150), BUDGET (default 15), STATE (default .t4-state)
set -eu
spent=${1:?spent so far}; dry=${2:-}
export AWS_PROFILE=${AWS_PROFILE:-mesh-916} AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
max_min=${MAX_MINUTES:-150}; budget=${BUDGET:-15}; state=${STATE:-.t4-state}; type=g4dn.xlarge
rate=$(aws pricing get-products --region us-east-1 --service-code AmazonEC2 \
  --filters Type=TERM_MATCH,Field=instanceType,Value=$type Type=TERM_MATCH,Field=location,Value='US East (N. Virginia)' \
  Type=TERM_MATCH,Field=operatingSystem,Value=Linux Type=TERM_MATCH,Field=tenancy,Value=Shared Type=TERM_MATCH,Field=preInstalledSw,Value=NA \
  Type=TERM_MATCH,Field=capacitystatus,Value=Used --output json | python3 -c "
import sys, json
p = json.loads(json.load(sys.stdin)['PriceList'][0])
print(next(iter(next(iter(p['terms']['OnDemand'].values()))['priceDimensions'].values()))['pricePerUnit']['USD'], p['publicationDate'])")
price=${rate% *}; pub=${rate#* }
volume=0.011                     # 100 GiB gp3 per hour
worst=$(python3 -c "print(round(($price + $volume) * ($max_min / 60 + 0.25), 2))")   # + 15 min boot/terminate slop
echo "rate \$$price/hr on-demand $type us-east-1 (Price List publication $pub) + volume \$$volume/hr; hard stop ${max_min} min"
echo "worst case \$$worst, spent so far \$$spent, budget \$$budget"
python3 -c "import sys; sys.exit(0 if $spent + $worst <= $budget else 1)" || { echo "refusing: over budget" >&2; exit 2; }
[ "$dry" = --dry-run ] && exit 0

mkdir -p "$state"; tag=hev-rerank-t4-$(date +%s)
ami=$(aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id --query Parameter.Value --output text)
vpc=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
myip=$(curl -s https://checkip.amazonaws.com)
aws ec2 create-key-pair --key-name $tag --query KeyMaterial --output text > "$state/key.pem"; chmod 600 "$state/key.pem"
sg=$(aws ec2 create-security-group --group-name $tag --description "hev-rerank T4 notebook run" --vpc-id $vpc --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $sg --protocol tcp --port 22 --cidr $myip/32 >/dev/null
iid=$(aws ec2 run-instances --image-id $ami --instance-type $type --key-name $tag --security-group-ids $sg \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$tag},{Key=owner,Value=hev-rerank-notebooks}]" \
  --user-data "#!/bin/bash
shutdown -h +$max_min" --query 'Instances[0].InstanceId' --output text)
printf 'tag=%s\nsg=%s\niid=%s\nlaunched=%s\nprice=%s\n' $tag $sg $iid $(date +%s) $price > "$state/env"
echo "launched $iid ($tag) at $(date -u +%FT%TZ); hard stop at $(date -u -d "+$max_min min" +%FT%TZ)"
aws ec2 wait instance-running --instance-ids $iid
ip=$(aws ec2 describe-instances --instance-ids $iid --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "ip=$ip" >> "$state/env"
for i in $(seq 1 60); do ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -i "$state/key.pem" ubuntu@$ip true 2>/dev/null && break; sleep 5; done
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$state/key.pem" ubuntu@$ip nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
