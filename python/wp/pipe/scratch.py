


from wp.pipe import Show, Asset, AssetBank


if __name__ == '__main__':
	# print("get tempest")
	# show = Show.get("tempest")
	#
	# print(show)
	# bank = AssetBank(show)


	bank = AssetBank("tempest")
	print("bank", bank)

	print(bank.allAssetDirs())

	dirA = bank.allAssetDirs()[0].parent
	assetTokens = Asset.tokensFromDirPath(dirA)
	print("assetTokens", assetTokens)
	asset = Asset(assetTokens)
	print("asset", asset)
	print(asset.diskPath())


