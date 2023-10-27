var _user_guide =
[
    [ "Interfaces", "_interfaces_.html", null ],
    [ "Client-side Programming", "_clientside.html", [
      [ "Making Remote Calls", "_clientside.html#MakingRemoteCalls", null ],
      [ "Client Stubs", "_clientside.html#ClientSide_ClientStubs", null ],
      [ "Transport Access", "_clientside.html#ClientSide_TransportAccess", null ],
      [ "Timeouts", "_clientside.html#Timeouts", null ],
      [ "Pinging", "_clientside.html#ClientStubs_Pinging", null ],
      [ "Client Progress Notification", "_clientside.html#ClientStubs_Progress", null ],
      [ "Remote Call Mode", "_clientside.html#ClientStubs_RemoteCallMode", null ],
      [ "Batched One-way Calls", "_clientside.html#BatchedOneWay", null ],
      [ "Custom Request and Response Data", "_clientside.html#UserData", null ],
      [ "Copy semantics", "_clientside.html#CopySemantics", null ]
    ] ],
    [ "Server-side Programming", "_serverside.html", [
      [ "Configuring a Server", "_serverside.html#Configure", [
        [ "Adding Transports", "_serverside.html#AddingTransports", null ],
        [ "Adding Servant Bindings", "_serverside.html#ServantBindings", null ],
        [ "Starting and Stopping a Server", "_serverside.html#StartingStopping", null ],
        [ "Server Threading", "_serverside.html#ServerThreading", null ]
      ] ],
      [ "Server-side Sessions", "_serverside.html#Sessions", [
        [ "Session Objects", "_serverside.html#SessionObjects", null ],
        [ "Custom Request and Response Data", "_serverside.html#CustomUserData", null ],
        [ "Session Information", "_serverside.html#SessionInformation", null ]
      ] ],
      [ "Access Control", "_serverside.html#AccessControl", null ],
      [ "Server Objects", "_serverside.html#ServerObjects", null ]
    ] ],
    [ "Serialization", "_serialization.html", [
      [ "Standard C++ Types", "_serialization.html#StandardCppTypes", null ],
      [ "User-defined Types", "_serialization.html#UserDefined", null ],
      [ "Binary Data", "_serialization.html#BinaryData", null ],
      [ "Portability", "_serialization.html#Portability", null ],
      [ "Serialization To and From Disk", "_serialization.html#StandAlone", null ]
    ] ],
    [ "Transports", "_transports.html", [
      [ "Transport Access", "_transports.html#Server", null ],
      [ "Transport Configuration", "_transports.html#Config", [
        [ "Maximum Incoming Message Lengths", "_transports.html#MaxMessageLength", null ],
        [ "Connection Limits", "_transports.html#ConnectionLimit", null ],
        [ "System Port Selection", "_transports.html#SystemPortSelection", null ],
        [ "IP-based Access Rules", "_transports.html#Access", null ],
        [ "IPv4/IPv6", "_transports.html#IPv6", null ],
        [ "Local Address Bindings for Clients", "_transports.html#Binding", null ],
        [ "Socket Level Access", "_transports.html#SocketAccess", null ]
      ] ],
      [ "Transport Implementations", "_transports.html#Implementations", [
        [ "TCP", "_transports.html#Tcp", null ],
        [ "UDP", "_transports.html#Udp", [
          [ "Multicasting", "_transports.html#Multicast", null ],
          [ "Broadcasting", "_transports.html#Broadcast", null ],
          [ "Address Sharing", "_transports.html#AddressSharing", null ],
          [ "Server Discovery", "_transports.html#ServerDiscovery", null ]
        ] ],
        [ "Win32 Named Pipes", "_transports.html#Win32Pipes", null ],
        [ "UNIX Domain Sockets", "_transports.html#Transports_LocalSockets", null ],
        [ "HTTP/HTTPS", "_transports.html#HTTP", [
          [ "Reverse proxies", "_transports.html#Reverse_proxies", null ]
        ] ]
      ] ]
    ] ],
    [ "Transport Protocols", "_transport_protocols.html", [
      [ "NTLM", "_transport_protocols.html#NTLM", null ],
      [ "Kerberos", "_transport_protocols.html#Kerberos", null ],
      [ "Negotiate", "_transport_protocols.html#Negotiate", null ],
      [ "SSL", "_transport_protocols.html#SSL", [
        [ "Schannel", "_transport_protocols.html#Schannel", null ],
        [ "OpenSSL", "_transport_protocols.html#OpenSSL", null ]
      ] ],
      [ "Compression", "_transport_protocols.html#Compression", null ]
    ] ],
    [ "Asynchronous Remote Calls", "_async_remote_calls.html", [
      [ "Asynchronous Invocation", "_async_remote_calls.html#ClientSideAsync", null ],
      [ "Asynchronous Dispatching", "_async_remote_calls.html#ServerSideAsync", null ]
    ] ],
    [ "Publish/Subscribe", "_pub_sub.html", [
      [ "Publishers", "_pub_sub.html#Publishers", null ],
      [ "Subscribers", "_pub_sub.html#Subscribers", null ],
      [ "Access Control", "_pub_sub.html#PubSubAccessControl", null ]
    ] ],
    [ "Proxy Endpoints", "_proxy_endpoints.html", [
      [ "Configuring a Proxy Server", "_proxy_endpoints.html#ProxyServer", null ],
      [ "Configuring a Destination Server", "_proxy_endpoints.html#DestinationServer", null ],
      [ "Connecting Through a Proxy Server", "_proxy_endpoints.html#ClientProxy", null ],
      [ "Two-way communication using proxy endpoints", "_proxy_endpoints.html#TwoWayCommunication", null ]
    ] ],
    [ "File Transfers", "_file_transfers.html", [
      [ "File Downloads", "_file_transfers.html#FileDownloads", null ],
      [ "File Uploads", "_file_transfers.html#FileUploads", null ],
      [ "Resuming File Transfers", "_file_transfers.html#ResumingFileTransfers", null ],
      [ "Monitoring File Transfers", "_file_transfers.html#MonitoringFileTransfers", null ],
      [ "Bandwidth Limits", "_file_transfers.html#BandwidthLimits", [
        [ "Server-Side Bandwidth Limits", "_file_transfers.html#RcfServerBandwidthLimit", null ],
        [ "Custom Server-Side Bandwidth Limits", "_file_transfers.html#CustomBandwidthLimit", null ],
        [ "Client-Side Bandwidth Limits", "_file_transfers.html#ClientBandwidthLimit", null ]
      ] ]
    ] ],
    [ "Versioning", "_versioning.html", [
      [ "Interface Versioning", "_versioning.html#InterfaceVersioning", [
        [ "Adding or Removing Methods", "_versioning.html#Methods", null ],
        [ "Adding or Removing Parameters", "_versioning.html#Parameters", null ],
        [ "Renaming Interfaces", "_versioning.html#Renaming", null ]
      ] ],
      [ "Archive Versioning", "_versioning.html#ArchiveVersioning", null ],
      [ "Runtime Versioning", "_versioning.html#RuntimeVersioning", [
        [ "Custom Version Negotiation", "_versioning.html#CustomVersionNegotiation", null ]
      ] ],
      [ "Protocol Buffers", "_versioning.html#ProtobufVersioning", null ]
    ] ],
    [ "Performance", "_performance.html", [
      [ "Remote calls", "_performance.html#PerfRemoteCalls", [
        [ "Zero Copy", "_performance.html#ZeroCopy", null ],
        [ "Zero Allocation", "_performance.html#ZeroAllocation", null ]
      ] ],
      [ "Object Caching", "_performance.html#Caching", null ],
      [ "Scalability", "_performance.html#Scalability", null ]
    ] ],
    [ "Advanced Serialization", "_advanced_serialization.html", [
      [ "Polymorphic Serialization", "_advanced_serialization.html#Polymorphic", null ],
      [ "Pointer Tracking", "_advanced_serialization.html#Pointers", null ],
      [ "Interchangeable Types", "_advanced_serialization.html#Interchangeable", null ],
      [ "Unicode Strings", "_advanced_serialization.html#Unicode", null ]
    ] ],
    [ "External Serialization", "_external_serialization.html", [
      [ "Protocol Buffers", "_external_serialization.html#Protobufs", [
        [ "Protocol Buffer Classes", "_external_serialization.html#ProtobufSerialization", null ],
        [ "Protocol Buffer Caching", "_external_serialization.html#ProtobufCaching", null ]
      ] ],
      [ "Boost.Serialization", "_external_serialization.html#BSer", null ]
    ] ],
    [ "Appendix - Logging", "_appendix_logging.html", null ],
    [ "Appendix - FAQ", "_appendix_faq.html", [
      [ "Building", "_appendix_faq.html#Building", null ],
      [ "Platforms", "_appendix_faq.html#Platforms", null ],
      [ "Programming", "_appendix_faq.html#Programming", null ],
      [ "Miscellaneous", "_appendix_faq.html#Misc", null ]
    ] ]
];